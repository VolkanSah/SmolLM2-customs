# =====================================================================================================
# Anti-Dump Algorithm (ADI)
# Copyright 2008 - 2025 S. Volkan Kücükbudak
# Apache License V2 + ESOL 1.1
# https://github.com/VolkanSah/Anti-Dump-Index
# =====================================================================================================

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import re
import numpy as np
import json
from pathlib import Path

@dataclass
class InputMetrics:
    noise: float
    effort: float
    context: float
    details: float
    bonus_factors: float
    penalty_factors: float
    repetition_penalty: float = 0.0

class DumpindexAnalyzer:
    def __init__(self, weights: Dict[str, float] = None, enable_logging: bool = False):
        self.weights = weights or {
            'noise': 1.0,
            'effort': 2.0,
            'context': 1.5,
            'details': 1.5,
            'bonus': 0.5,
            'penalty': 1.0
        }
        self.enable_logging = enable_logging
        self.log_file = Path('adi_logs.jsonl')

        self.noise_patterns = {
            'urgency': r'\b(urgent|asap|emergency|!!+|\?\?+)\b',
            'informal': r'\b(pls|plz|thx|omg|wtf)\b',
            'vague': r'\b(something|somehow|maybe|probably)\b'
        }
        self.detail_patterns = {
            'code_elements': r'\b(function|class|method|variable|array|object|def|return)\b',
            'technical_terms': r'\b(error|exception|bug|issue|crash|fail|traceback|stack)\b',
            'specifics': r'[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*'
        }
        self.context_indicators = {
            'background': r'\b(because|since|as|when|while)\b',
            'environment': r'\b(using|version|environment|platform|system)\b',
            'goal': r'\b(trying to|want to|need to|goal is|attempting to)\b'
        }

    def _has_negation_before(self, text: str, match_pos: int, window_size: int = 50) -> bool:
        window_start = max(0, match_pos - window_size)
        window = text[window_start:match_pos].lower()
        return bool(re.search(r'\b(no|not|never|without|dont|don\'t|doesnt|doesn\'t)\b', window))

    def calculate_repetition_penalty(self, text: str) -> float:
        words = text.lower().split()
        if len(words) == 0:
            return 0.0
        unique_ratio = len(set(words)) / len(words)
        word_counts = {}
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
        max_repetition = max(word_counts.values()) if word_counts else 1
        repetition_factor = min(max_repetition / len(words), 0.5)
        penalty = (1 - unique_ratio) * 2 + repetition_factor * 2
        return min(penalty, 3.0)

    def calculate_noise(self, text: str) -> Tuple[float, Dict]:
        noise_count = 0
        noise_details = {}
        for category, pattern in self.noise_patterns.items():
            matches = re.findall(pattern, text.lower())
            noise_count += len(matches)
            noise_details[category] = matches
        total_words = len(text.split())
        return (noise_count / max(total_words, 1), noise_details)

    def calculate_effort(self, text: str) -> float:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences:
            return 0.0
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        has_formatting = bool(re.search(r'```|\*\*|\n\s*\n', text))
        has_punctuation = bool(re.search(r'[.,;:]', text))
        sentence_quality = (
            (len(sentences) >= 3) * 1.0 +
            (20 <= avg_sentence_length <= 50) * 2.0 +
            (avg_sentence_length >= 5) * 0.5
        )
        return min(5.0, sentence_quality + has_formatting * 1.5 + has_punctuation * 1.5)

    def calculate_context(self, text: str) -> float:
        context_score = 0.0
        for category, pattern in self.context_indicators.items():
            for match in re.finditer(pattern, text.lower()):
                if not self._has_negation_before(text, match.start()):
                    context_score += 1.0
                    break
        return min(5.0, context_score)

    def calculate_details(self, text: str) -> Tuple[float, Dict]:
        detail_score = 0.0
        detail_findings = {}
        for category, pattern in self.detail_patterns.items():
            matches = re.findall(pattern, text.lower())
            score = len(matches) * 0.5
            detail_findings[category] = matches
            detail_score += score
        return (min(5.0, detail_score), detail_findings)

    def calculate_bonus_factors(self, text: str) -> float:
        bonus_score = 0.0
        if re.search(r'```[\s\S]*?```', text):
            bonus_score += 1.0
        if re.search(r'\[.*?\]\(.*?\)', text):
            bonus_score += 0.5
        if re.search(r'\n\s*[-*+]\s', text):
            bonus_score += 0.5
        return bonus_score

    def calculate_penalty_factors(self, text: str) -> Tuple[float, Dict]:
        penalties = {}
        alpha_chars = re.findall(r'[a-zA-Z]', text)
        if alpha_chars:
            caps_ratio = len(re.findall(r'[A-Z]', text)) / len(alpha_chars)
            if caps_ratio > 0.7:
                penalties['excessive_caps'] = caps_ratio
        excessive_punctuation = len(re.findall(r'[!?]{2,}', text))
        if excessive_punctuation:
            penalties['excessive_punctuation'] = excessive_punctuation
        if len(text.split()) < 10:
            penalties['too_short'] = 1.0
        penalty_score = sum(penalties.values()) if penalties else 0
        return (min(5.0, penalty_score), penalties)

    def calculate_adi(self, metrics: InputMetrics) -> float:
        try:
            numerator = (
                self.weights['noise'] * metrics.noise -
                (self.weights['effort'] * metrics.effort +
                 self.weights['bonus'] * metrics.bonus_factors)
            )
            denominator = (
                self.weights['context'] * metrics.context +
                self.weights['details'] * metrics.details +
                self.weights['penalty'] * metrics.penalty_factors +
                metrics.repetition_penalty
            )
            return numerator / max(denominator, 0.1)
        except Exception as e:
            return float('inf')

    def analyze_input(self, text: str, user_context: Optional[Dict] = None) -> Dict:
        noise_value, noise_details = self.calculate_noise(text)
        effort_value = self.calculate_effort(text)
        context_value = self.calculate_context(text)
        details_value, detail_findings = self.calculate_details(text)
        bonus_value = self.calculate_bonus_factors(text)
        penalty_value, penalty_details = self.calculate_penalty_factors(text)
        repetition_value = self.calculate_repetition_penalty(text)

        metrics = InputMetrics(
            noise=noise_value, effort=effort_value, context=context_value,
            details=details_value, bonus_factors=bonus_value,
            penalty_factors=penalty_value, repetition_penalty=repetition_value
        )
        adi = self.calculate_adi(metrics)

        adi_adjusted = adi
        if user_context:
            if user_context.get('tier') == 'enterprise':
                adi_adjusted *= 0.8
            if user_context.get('history_avg', 0) < 0:
                adi_adjusted *= 0.9

        decision = self._make_decision(adi_adjusted)
        recommendations = self._generate_recommendations(
            metrics, noise_details, detail_findings, penalty_details
        )

        return {
            'adi': round(adi, 3),
            'adi_adjusted': round(adi_adjusted, 3) if user_context else None,
            'metrics': {
                'noise': round(noise_value, 3), 'effort': round(effort_value, 3),
                'context': round(context_value, 3), 'details': round(details_value, 3),
                'bonus_factors': round(bonus_value, 3), 'penalty_factors': round(penalty_value, 3),
                'repetition_penalty': round(repetition_value, 3)
            },
            'decision': decision,
            'recommendations': recommendations,
            'details': {
                'noise_findings': noise_details,
                'technical_details': detail_findings,
                'penalties': penalty_details
            }
        }

    def _make_decision(self, adi: float) -> str:
        if adi > 1:
            return "REJECT"
        elif 0 <= adi <= 1:
            return "MEDIUM_PRIORITY"
        else:
            return "HIGH_PRIORITY"

    def _generate_recommendations(self, metrics, noise_details, detail_findings, penalty_details):
        recommendations = []
        if metrics.noise > 0.3:
            recommendations.append("Reduce informal or urgent expressions.")
        if metrics.context < 1.0:
            recommendations.append("Provide more context (environment, background, goal).")
        if metrics.details < 1.0:
            recommendations.append("Include specific technical details or error messages.")
        if metrics.effort < 2.0:
            recommendations.append("Improve the structure of your input with proper sentences.")
        if metrics.repetition_penalty > 1.0:
            recommendations.append("Avoid repeating the same keywords excessively.")
        if metrics.penalty_factors > 0:
            if 'excessive_caps' in penalty_details:
                recommendations.append("Avoid excessive capitalization.")
            if 'excessive_punctuation' in penalty_details:
                recommendations.append("Reduce excessive punctuation marks.")
            if 'too_short' in penalty_details:
                recommendations.append("Provide a more detailed description (minimum 10 words).")
        if not recommendations:
            recommendations.append("Your input quality is excellent. No improvements needed.")
        return recommendations

    def _log_analysis(self, text: str, adi: float, metrics: InputMetrics):
        log_entry = {
            'text_hash': hash(text), 'text_length': len(text), 'adi': round(adi, 3),
            'metrics': {
                'noise': round(metrics.noise, 3), 'effort': round(metrics.effort, 3),
                'context': round(metrics.context, 3), 'details': round(metrics.details, 3),
                'bonus_factors': round(metrics.bonus_factors, 3),
                'penalty_factors': round(metrics.penalty_factors, 3),
                'repetition_penalty': round(metrics.repetition_penalty, 3)
            }
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
