#!/usr/bin/env python3
"""
Proper Nomic Game - Enhanced with Model Performance Tracking
- Comprehensive behavioral analysis and coherence tracking
- Cross-game model performance metrics and weighted assignment
- Anti-tie-breaking obsession system with semantic analysis
- Strategic engagement detection for quality control
- Multi-dimensional model scoring and classification
"""

import os
import json
import time
import random
import re
import threading
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
from flask import Flask, render_template_string, jsonify, request
import requests

@dataclass
class RuleEffect:
    """Parsed rule effects that can be executed"""
    trigger: str  # "turn_start", "vote_pass", "vote_fail", "dice_roll", etc.
    condition: Optional[str] = None  # "points < 50", "roll == 6", etc.
    action: str = ""  # "add_points", "subtract_points", "steal_points", etc.
    target: str = "current_player"  # "current_player", "all_players", "proposer", etc.
    value: int = 0
    description: str = ""

@dataclass
class ParsedRule:
    """A rule with parsed executable effects"""
    id: int
    text: str
    mutable: bool
    effects: List[RuleEffect] = field(default_factory=list)
    author: Optional[int] = None
    turn_added: Optional[int] = None

@dataclass
class ModelMetrics:
    """Comprehensive model performance and behavioral metrics"""
    model_name: str
    
    # Performance Metrics
    total_games: int = 0
    proposals_made: int = 0
    proposals_passed: int = 0
    total_points_gained: int = 0
    games_won: int = 0
    
    # Coherence Metrics
    coherence_scores: List[float] = field(default_factory=list)
    syntax_errors: int = 0
    validation_failures: int = 0
    logic_inconsistencies: int = 0
    
    # Memory & Learning Metrics  
    memory_reference_count: int = 0
    learning_evidence_count: int = 0
    repeated_failures: int = 0
    adaptation_score: float = 0.0
    
    # Engagement Metrics
    strategic_depth_scores: List[float] = field(default_factory=list)
    game_understanding_scores: List[float] = field(default_factory=list)
    random_behavior_flags: int = 0
    
    # Error Analysis
    error_types: Dict[str, int] = field(default_factory=dict)
    correction_responses: int = 0
    improvement_trend: float = 0.0
    
    # Tie-Breaking Obsession Tracking
    tie_related_proposals: int = 0
    proposal_diversity_scores: List[float] = field(default_factory=list)
    
    # Timestamps for decay calculation
    last_updated: str = ""
    performance_history: List[Dict] = field(default_factory=list)
    
    def calculate_success_rate(self) -> float:
        return (self.proposals_passed / max(self.proposals_made, 1)) * 100
    
    def calculate_average_coherence(self) -> float:
        return sum(self.coherence_scores) / max(len(self.coherence_scores), 1)
    
    def calculate_strategic_engagement(self) -> float:
        strategic_avg = sum(self.strategic_depth_scores) / max(len(self.strategic_depth_scores), 1)
        understanding_avg = sum(self.game_understanding_scores) / max(len(self.game_understanding_scores), 1)
        return (strategic_avg + understanding_avg) / 2
    
    def calculate_overall_score(self) -> float:
        """Calculate weighted overall performance score (0-100)"""
        success_rate = self.calculate_success_rate()
        coherence = self.calculate_average_coherence()
        engagement = self.calculate_strategic_engagement()
        diversity = sum(self.proposal_diversity_scores) / max(len(self.proposal_diversity_scores), 1)
        
        # Error penalty
        error_penalty = min(20, (self.syntax_errors + self.validation_failures) * 2)
        tie_penalty = min(15, self.tie_related_proposals * 3)
        
        score = (success_rate * 0.3 + coherence * 0.25 + engagement * 0.25 + diversity * 0.2) - error_penalty - tie_penalty
        return max(0, min(100, score))

@dataclass
class Player:
    id: int
    name: str
    role: str
    points: int = 50
    model: str = "llama3.2:3b"
    port: int = 11434
    assigned_model_metrics: Optional[ModelMetrics] = None

@dataclass  
class Proposal:
    id: int
    player_id: int
    rule_text: str
    explanation: str
    internal_thoughts: str
    turn: int
    votes: Dict[int, bool] = None
    
    def __post_init__(self):
        if self.votes is None:
            self.votes = {}

class ModelPerformanceManager:
    """Cross-game model performance tracking and management"""
    
    def __init__(self, storage_path: str = "model_performance.json"):
        self.storage_path = storage_path
        self.metrics: Dict[str, ModelMetrics] = {}
        self.load_metrics()
        
        # Tie-breaking keywords for detection
        self.tie_keywords = {
            'tie', 'tied', 'equal', 'same', 'break', 'breaker', 'tiebreak', 
            'deadlock', 'even', 'identical', 'resolve', 'determine', 'winner'
        }
        
        # Strategic keywords for depth analysis
        self.strategic_keywords = {
            'strategy', 'tactical', 'advantage', 'position', 'opportunity', 'leverage',
            'timing', 'future', 'long-term', 'consequence', 'alliance', 'cooperation'
        }
        
    def load_metrics(self):
        """Load performance metrics from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for model_name, metrics_data in data.items():
                        # Convert back to ModelMetrics object
                        metrics = ModelMetrics(**metrics_data)
                        self.metrics[model_name] = metrics
            except Exception as e:
                print(f"Error loading metrics: {e}")
                self.metrics = {}
                
    def save_metrics(self):
        """Save performance metrics to storage"""
        try:
            data = {}
            for model_name, metrics in self.metrics.items():
                data[model_name] = asdict(metrics)
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics: {e}")
            
    def get_or_create_metrics(self, model_name: str) -> ModelMetrics:
        """Get existing metrics or create new ones for a model"""
        if model_name not in self.metrics:
            self.metrics[model_name] = ModelMetrics(model_name=model_name)
        return self.metrics[model_name]
        
    def analyze_proposal_coherence(self, rule_text: str, explanation: str, internal_thought: str) -> float:
        """Analyze coherence of a proposal (0-100 score)"""
        score = 50.0  # Base score
        
        # Text clarity and completeness
        if len(rule_text.strip()) < 10:
            score -= 20
        if len(explanation.strip()) < 20:
            score -= 15
        if len(internal_thought.strip()) < 10:
            score -= 10
            
        # Check for syntax issues
        if rule_text.count('.') == 0:  # No ending punctuation
            score -= 5
        if not rule_text[0].isupper():  # Doesn't start with capital
            score -= 5
            
        # Check for logical consistency between internal thought and explanation
        thought_words = set(internal_thought.lower().split())
        explanation_words = set(explanation.lower().split())
        overlap = len(thought_words.intersection(explanation_words))
        consistency_bonus = min(15, overlap * 2)
        score += consistency_bonus
        
        # Check for strategic depth
        strategic_words = sum(1 for word in rule_text.lower().split() if word in self.strategic_keywords)
        score += min(10, strategic_words * 3)
        
        return max(0, min(100, score))
        
    def analyze_memory_usage(self, internal_thought: str, explanation: str, game_context: str) -> float:
        """Analyze evidence of memory and learning (0-100 score)"""
        score = 0.0
        text_combined = (internal_thought + " " + explanation).lower()
        
        # Look for references to past events
        memory_indicators = ['previous', 'last', 'before', 'earlier', 'remember', 'learned', 'failed', 'worked']
        memory_refs = sum(1 for indicator in memory_indicators if indicator in text_combined)
        score += min(30, memory_refs * 5)
        
        # Look for specific game references
        if 'turn' in text_combined and any(char.isdigit() for char in text_combined):
            score += 20
        if 'rule' in text_combined and any(char.isdigit() for char in text_combined):
            score += 15
            
        # Check for learning evidence
        learning_words = ['learned', 'noticed', 'realized', 'understand', 'see that', 'because']
        learning_evidence = sum(1 for word in learning_words if word in text_combined)
        score += min(25, learning_evidence * 8)
        
        return min(100, score)
        
    def detect_tie_obsession(self, rule_text: str, explanation: str, internal_thought: str) -> bool:
        """Detect if proposal is obsessed with tie-breaking"""
        text_combined = (rule_text + " " + explanation + " " + internal_thought).lower()
        
        tie_word_count = sum(1 for word in self.tie_keywords if word in text_combined)
        word_count = len(text_combined.split())
        
        # If more than 3% of words are tie-related, flag as obsessed
        return tie_word_count > 0 and (tie_word_count / word_count) > 0.03
        
    def calculate_proposal_diversity(self, rule_text: str, all_previous_proposals: List[str]) -> float:
        """Calculate diversity score against all previous proposals (0-100)"""
        if not all_previous_proposals:
            return 100.0
            
        current_words = set(rule_text.lower().split())
        similarities = []
        
        for prev_proposal in all_previous_proposals:
            prev_words = set(prev_proposal.lower().split())
            if len(current_words) == 0 or len(prev_words) == 0:
                continue
                
            # Calculate Jaccard similarity
            intersection = len(current_words.intersection(prev_words))
            union = len(current_words.union(prev_words))
            similarity = intersection / union if union > 0 else 0
            similarities.append(similarity)
            
        if not similarities:
            return 100.0
            
        # Diversity = 100 - (max similarity * 100)
        max_similarity = max(similarities)
        return max(0, 100 - (max_similarity * 100))
        
    def analyze_strategic_engagement(self, rule_text: str, explanation: str, internal_thought: str, game_state: Dict) -> float:
        """Analyze strategic engagement and game understanding (0-100)"""
        score = 0.0
        
        # Strategic depth analysis
        strategic_word_count = 0
        all_text = (rule_text + " " + explanation + " " + internal_thought).lower()
        for word in self.strategic_keywords:
            strategic_word_count += all_text.count(word)
        score += min(30, strategic_word_count * 5)
        
        # Game state awareness
        if 'points' in all_text:
            score += 15
        if 'vote' in all_text or 'voting' in all_text:
            score += 10
        if 'rule' in all_text:
            score += 10
            
        # Complexity and thoughtfulness
        if len(internal_thought.split()) > 15:
            score += 15
        if len(explanation.split()) > 10:
            score += 10
            
        # Check for goal-oriented thinking
        goal_words = ['win', 'winning', 'advantage', 'benefit', 'help', 'improve', 'gain']
        goal_thinking = sum(1 for word in goal_words if word in all_text)
        score += min(20, goal_thinking * 4)
        
        return min(100, score)
        
    def update_model_performance(self, model_name: str, proposal_data: Dict, game_outcome: Dict):
        """Update comprehensive performance metrics for a model"""
        metrics = self.get_or_create_metrics(model_name)
        
        # Basic performance updates
        metrics.proposals_made += 1
        if proposal_data.get('passed', False):
            metrics.proposals_passed += 1
            
        # Coherence analysis
        coherence_score = self.analyze_proposal_coherence(
            proposal_data['rule_text'],
            proposal_data['explanation'], 
            proposal_data['internal_thought']
        )
        metrics.coherence_scores.append(coherence_score)
        
        # Memory analysis
        memory_score = self.analyze_memory_usage(
            proposal_data['internal_thought'],
            proposal_data['explanation'],
            proposal_data.get('game_context', '')
        )
        if memory_score > 50:
            metrics.memory_reference_count += 1
            
        # Strategic engagement
        engagement_score = self.analyze_strategic_engagement(
            proposal_data['rule_text'],
            proposal_data['explanation'],
            proposal_data['internal_thought'],
            proposal_data.get('game_state', {})
        )
        metrics.strategic_depth_scores.append(engagement_score)
        
        # Tie obsession detection
        if self.detect_tie_obsession(
            proposal_data['rule_text'],
            proposal_data['explanation'],
            proposal_data['internal_thought']
        ):
            metrics.tie_related_proposals += 1
            
        # Diversity scoring
        diversity_score = self.calculate_proposal_diversity(
            proposal_data['rule_text'],
            proposal_data.get('all_previous_proposals', [])
        )
        metrics.proposal_diversity_scores.append(diversity_score)
        
        # Error tracking
        if proposal_data.get('validation_failed', False):
            metrics.validation_failures += 1
        if proposal_data.get('syntax_error', False):
            metrics.syntax_errors += 1
            
        # Update timestamp
        metrics.last_updated = datetime.now().isoformat()
        
        # Save metrics
        self.save_metrics()
        
    def get_weighted_model_selection(self, available_models: List[str]) -> List[str]:
        """Return models weighted by performance for random selection"""
        if not available_models:
            return available_models
            
        weights = []
        for model in available_models:
            if model in self.metrics:
                score = self.metrics[model].calculate_overall_score()
                # Convert score to weight (minimum weight of 10 for recovery)
                weight = max(10, score)
            else:
                # New models get average weight
                weight = 50
            weights.append(weight)
            
        # Create weighted list
        weighted_models = []
        for i, model in enumerate(available_models):
            count = int(weights[i] / 10)  # Scale down weights
            weighted_models.extend([model] * max(1, count))
            
        return weighted_models
        
    def get_model_statistics(self) -> Dict:
        """Get comprehensive statistics for all models"""
        stats = {}
        for model_name, metrics in self.metrics.items():
            stats[model_name] = {
                'overall_score': metrics.calculate_overall_score(),
                'success_rate': metrics.calculate_success_rate(),
                'coherence': metrics.calculate_average_coherence(),
                'engagement': metrics.calculate_strategic_engagement(),
                'games_played': metrics.total_games,
                'proposals_made': metrics.proposals_made,
                'tie_obsession': metrics.tie_related_proposals,
                'error_count': metrics.syntax_errors + metrics.validation_failures,
                'last_updated': metrics.last_updated
            }
        return stats

class DeliberationManager:
    """Manages internal deliberation loops for strategic thinking"""
    
    def __init__(self):
        # Define 8 distinct proposal categories for diversity enforcement
        self.proposal_categories = {
            'point_distribution': {
                'keywords': ['points', 'score', 'gain', 'lose', 'award', 'penalty', 'distribute'],
                'description': 'Rules about how points are gained, lost, or distributed',
                'examples': ['bonus points for voting', 'point penalties for failed proposals', 'point redistribution']
            },
            'turn_mechanics': {
                'keywords': ['turn', 'order', 'skip', 'extra', 'sequence', 'round'],
                'description': 'Rules about turn order, timing, or turn-taking mechanics',
                'examples': ['change turn order', 'extra turns', 'skip turns', 'simultaneous actions']
            },
            'victory_conditions': {
                'keywords': ['win', 'victory', 'goal', 'target', 'achieve', 'finish'],
                'description': 'Rules that modify winning conditions or game end states',
                'examples': ['new victory conditions', 'multiple win paths', 'victory point modifications']
            },
            'player_interaction': {
                'keywords': ['alliance', 'team', 'cooperation', 'communication', 'negotiate', 'share'],
                'description': 'Rules about player communication, alliances, or cooperation',
                'examples': ['alliance formation', 'information sharing', 'joint proposals']
            },
            'resource_management': {
                'keywords': ['resource', 'currency', 'tokens', 'cards', 'items', 'assets'],
                'description': 'Rules introducing resources, currencies, or assets to manage',
                'examples': ['virtual currencies', 'resource collection', 'item trading']
            },
            'information_systems': {
                'keywords': ['secret', 'hidden', 'reveal', 'information', 'knowledge', 'private'],
                'description': 'Rules about information visibility, secrets, or knowledge sharing',
                'examples': ['hidden information', 'secret voting', 'information reveals']
            },
            'penalty_mechanisms': {
                'keywords': ['penalty', 'punishment', 'consequence', 'fine', 'restriction', 'limit'],
                'description': 'Rules that create penalties or restrictions for certain actions',
                'examples': ['voting penalties', 'proposal restrictions', 'action limits']
            },
            'meta_rules': {
                'keywords': ['rule', 'vote', 'voting', 'proposal', 'change', 'modify'],
                'description': 'Rules about how rules are created, modified, or voted on',
                'examples': ['voting requirements', 'proposal formats', 'rule modification']
            }
        }
        
        self.recent_categories = []  # Track recently used categories
        self.category_usage_history = {cat: 0 for cat in self.proposal_categories.keys()}
        
    def generate_nomic_context_header(self, game_state: Dict, player=None) -> str:
        """Generate explicit Nomic game context header for every prompt"""
        sorted_players = sorted(game_state['players'], key=lambda p: p['points'], reverse=True)
        
        # Calculate points needed to win for each player
        player_progress = []
        for i, p in enumerate(sorted_players):
            points_needed = max(0, 100 - p['points'])
            rank_indicator = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
            player_progress.append(f"{rank_indicator} {p['name']}: {p['points']}/100 pts (need {points_needed})")
        
        # Get current leader info
        leader = sorted_players[0]
        leader_advantage = leader['points'] - sorted_players[1]['points'] if len(sorted_players) > 1 else 0
        
        # Player-specific context
        player_context = ""
        if player:
            my_rank = next(i for i, p in enumerate(sorted_players, 1) if p['name'] == player.name)
            my_points = player.points
            points_to_win = 100 - my_points
            player_context = f"""
🎯 YOUR STATUS: Rank #{my_rank} with {my_points}/100 points ({points_to_win} points needed to WIN)
📈 YOUR GOAL: Get to 100 points before anyone else does"""
        
        context_header = f"""
═══════════════════════════════════════════════════════════════
🎮 GAME: NOMIC (Self-Modifying Competitive Rule Game)
🏆 VICTORY CONDITION: First player to reach 100 POINTS wins the game
🎯 WIN OBJECTIVE: Propose and vote strategically to reach 100 points first
═══════════════════════════════════════════════════════════════

📊 CURRENT GAME STATUS (Turn {game_state.get('turn', 1)}):
{chr(10).join(player_progress)}

🏁 VICTORY ANALYSIS:
• Current Leader: {leader['name']} with {leader['points']} points (ahead by {leader_advantage})
• Closest to Victory: {leader['name']} needs {100 - leader['points']} more points to WIN
• Game Situation: {"Early stage" if leader['points'] < 70 else "LATE STAGE - Victory approaching!" if leader['points'] < 90 else "CRITICAL - Someone about to WIN!"}
{player_context}

📜 NOMIC RULES PRIMER:
• NOMIC is a game where players compete by changing the rules to help themselves win
• You propose RULES that affect: point scoring, turn order, voting, player interactions
• Valid Nomic rules affect GAMEPLAY: how points are gained/lost, how turns work, how voting works
• INVALID rules: Real-world topics (city planning, energy audits, external policies)

✅ EXAMPLES OF VALID NOMIC RULES:
• "Players in last place gain 3 bonus points each turn"
• "Players may steal 2 points from opponents who vote against their proposals"  
• "The current leader loses 1 point at the start of each turn"
• "Players may skip their turn to give another player -2 points"
• "Players who propose rules in new categories get +2 bonus points"

❌ INVALID (NOT NOMIC RULES):
• City council policies, energy audits, building regulations
• Real-world governance, environmental policies
• Rules about external games or non-Nomic activities

🎯 STRATEGIC REALITY:
• You are competing to WIN - reach 100 points first
• Other players are your COMPETITORS trying to beat you
• Propose rules that help YOU more than others
• Vote for rules that benefit YOU, against rules that benefit opponents more
═══════════════════════════════════════════════════════════════
"""
        return context_header
        
    def categorize_proposal(self, rule_text: str, explanation: str) -> str:
        """Categorize a proposal into one of the 8 categories"""
        combined_text = (rule_text + " " + explanation).lower()
        
        category_scores = {}
        for category, data in self.proposal_categories.items():
            score = 0
            for keyword in data['keywords']:
                score += combined_text.count(keyword)
            category_scores[category] = score
            
        # Return category with highest score, or 'meta_rules' as default
        if max(category_scores.values()) == 0:
            return 'meta_rules'
        return max(category_scores, key=category_scores.get)
        
    def is_category_overused(self, category: str, turns_since_last: int = 3) -> bool:
        """Check if a category has been used too recently"""
        if len(self.recent_categories) < turns_since_last:
            return False
        return category in self.recent_categories[-turns_since_last:]
        
    def get_underused_categories(self) -> List[str]:
        """Get categories that haven't been used much"""
        min_usage = min(self.category_usage_history.values())
        return [cat for cat, usage in self.category_usage_history.items() if usage == min_usage]
        
    def record_category_usage(self, category: str):
        """Record that a category has been used"""
        self.recent_categories.append(category)
        self.recent_categories = self.recent_categories[-10:]  # Keep last 10
        self.category_usage_history[category] += 1
        
    def generate_deliberation_prompts(self, player, game_state: Dict, deliberation_turn: int) -> str:
        """Generate prompts for each deliberation turn"""
        
        if deliberation_turn == 1:
            return self._generate_state_analysis_prompt(player, game_state)
        elif deliberation_turn == 2:
            return self._generate_gap_analysis_prompt(player, game_state)
        elif deliberation_turn == 3:
            return self._generate_category_selection_prompt(player, game_state)
        elif deliberation_turn == 4:
            return self._generate_impact_modeling_prompt(player, game_state)
        elif deliberation_turn == 5:
            return self._generate_final_selection_prompt(player, game_state)
        else:
            return self._generate_final_selection_prompt(player, game_state)
            
    def _generate_state_analysis_prompt(self, player, game_state: Dict) -> str:
        """Turn 1: Analyze current position and threats"""
        context_header = self.generate_nomic_context_header(game_state, player)
        
        return f"""{context_header}

🧠 DELIBERATION TURN 1: STRATEGIC POSITION ANALYSIS

Your Role in this NOMIC game: {player.role}

STRATEGIC ANALYSIS REQUIRED:
1. Who is currently winning and by how much in this 100-point race?
2. Who poses the biggest threat to YOUR victory in reaching 100 points first?
3. What is your current strategic position (leading/trailing/middle)?
4. Which players might help you vs block you from reaching 100 points?
5. What patterns do you see in recent Nomic rule changes?

OUTPUT FORMAT:
THREAT_ASSESSMENT: [Identify your biggest threats to winning this Nomic game]
POSITION_ANALYSIS: [Your current strategic situation in the 100-point race]
ALLIANCE_POTENTIAL: [Who might work with you vs against you in Nomic]
PATTERN_OBSERVATION: [What trends do you see in the Nomic rule changes]

🎯 REMEMBER: You're playing NOMIC to reach 100 points first. Focus on YOUR chances of winning this specific game."""
        
    def _generate_gap_analysis_prompt(self, player, game_state: Dict) -> str:
        """Turn 2: Identify strategic gaps and opportunities"""
        context_header = self.generate_nomic_context_header(game_state, player)
        
        return f"""{context_header}

🧠 DELIBERATION TURN 2: NOMIC RULE GAP ANALYSIS

Your Role in this NOMIC game: {player.role}
Current Nomic Game Turn: {game_state.get('turn', 1)}

EXISTING NOMIC RULES ANALYSIS:
{chr(10).join([f"Nomic Rule {r['id']}: {r['text']}" for r in game_state.get('mutable_rules', [])])}

STRATEGIC GAP IDENTIFICATION IN NOMIC:
1. What Nomic game mechanics are completely unexplored?
2. What aspects of point scoring in Nomic are unaddressed?
3. Are there Nomic rule loopholes you could exploit to reach 100 points?
4. What Nomic areas give you the best advantage given your role?

NOMIC CATEGORY OPPORTUNITIES:
Based on your analysis, which Nomic rule categories offer the most strategic potential for reaching 100 points:

- Point Distribution: {self.proposal_categories['point_distribution']['description']} in Nomic
- Turn Mechanics: {self.proposal_categories['turn_mechanics']['description']} in Nomic  
- Victory Conditions: {self.proposal_categories['victory_conditions']['description']} in Nomic
- Player Interaction: {self.proposal_categories['player_interaction']['description']} in Nomic
- Resource Management: {self.proposal_categories['resource_management']['description']} in Nomic
- Information Systems: {self.proposal_categories['information_systems']['description']} in Nomic
- Penalty Mechanisms: {self.proposal_categories['penalty_mechanisms']['description']} in Nomic

OUTPUT FORMAT:
UNEXPLORED_MECHANICS: [What Nomic mechanics are missing from the game]
EXPLOITABLE_GAPS: [Nomic opportunities you could use to reach 100 points]
TOP_CATEGORIES: [Which 2-3 Nomic categories offer you the most advantage]
ROLE_ADVANTAGE: [How your role gives you unique Nomic opportunities]

🎯 Think competitively about NOMIC - what Nomic rules help YOU reach 100 points first, not everyone."""
        
    def _generate_category_selection_prompt(self, player, game_state: Dict) -> str:
        """Turn 3: Select specific category and approach"""
        context_header = self.generate_nomic_context_header(game_state, player)
        underused = self.get_underused_categories()
        overused = [cat for cat in self.proposal_categories.keys() if self.is_category_overused(cat)]
        
        return f"""{context_header}

🧠 DELIBERATION TURN 3: NOMIC CATEGORY SELECTION & STRATEGIC APPROACH

Your Role in this NOMIC game: {player.role}
Your Strategic Position: {game_state.get('position_analysis', 'Unknown')}

NOMIC CATEGORY AVAILABILITY ANALYSIS:
UNDERUSED NOMIC CATEGORIES (Recommended): {', '.join(underused)}
OVERUSED NOMIC CATEGORIES (Avoid): {', '.join(overused)}

STRATEGIC NOMIC CATEGORY EVALUATION:
For each available Nomic category, consider:

1. POINT DISTRIBUTION SYSTEMS (in Nomic):
   - Could you create Nomic rules that favor your position?
   - Are there point-scoring opportunities others missed in Nomic?

2. TURN MECHANICS (in Nomic):
   - Could you gain extra turns or disrupt others' turns in Nomic?
   - Are there timing advantages you could exploit in Nomic?

3. VICTORY CONDITIONS (in Nomic):
   - Could you create alternative Nomic win conditions that favor you?
   - Are there shortcuts to 100 points others haven't considered?

4. PLAYER INTERACTION (in Nomic):
   - Could you force alliances that benefit you in Nomic?
   - Are there Nomic cooperation rules that help your position?

5. RESOURCE MANAGEMENT (in Nomic):
   - Could you introduce Nomic resources you'd be good at managing?
   - Are there scarcity mechanics that favor your Nomic strategy?

6. INFORMATION SYSTEMS (in Nomic):
   - Could you create information advantages for yourself in Nomic?
   - Are there secret Nomic mechanisms that help your role?

7. PENALTY MECHANISMS (in Nomic):
   - Could you penalize Nomic behaviors that hurt your chances?
   - Are there Nomic restrictions that favor your playstyle?

OUTPUT FORMAT:
SELECTED_CATEGORY: [Your chosen Nomic category]
STRATEGIC_RATIONALE: [Why this Nomic category gives YOU the best advantage]
COMPETITIVE_ANGLE: [How this Nomic rule hurts your competitors or helps your position]
SPECIFIC_APPROACH: [The general type of Nomic rule you'll propose]

🎯 Choose based on YOUR winning chances in NOMIC, not fairness to others."""
        
    def _generate_impact_modeling_prompt(self, player, game_state: Dict) -> str:
        """Turn 4: Model impacts of potential approaches"""
        context_header = self.generate_nomic_context_header(game_state, player)
        
        return f"""{context_header}

🧠 DELIBERATION TURN 4: NOMIC IMPACT MODELING & STRATEGIC OPTIMIZATION

Your Selected Nomic Category: {game_state.get('selected_category', 'Unknown')}
Your Current Position: #{game_state.get('rank', '?')} with {player.points} points in the 100-point race

NOMIC IMPACT MODELING EXERCISE:
For your chosen Nomic approach, analyze the following impacts:

IMPACT ON YOU IN NOMIC:
- How many points could you potentially gain in this Nomic game?
- How does this improve your winning chances to reach 100 points?
- What strategic advantages does this give you in Nomic?
- Are there any risks or downsides for you in this Nomic rule?

IMPACT ON COMPETITORS IN NOMIC:
- Who gets hurt most by this Nomic rule?
- Who benefits least from this Nomic rule?
- Will this Nomic rule slow down the current leader?
- Does this create problems for your biggest threats in the 100-point race?

VOTING LIKELIHOOD ANALYSIS FOR NOMIC:
- Which players have incentive to vote FOR this Nomic rule?
- Which players have incentive to vote AGAINST this Nomic rule?
- How can you frame this Nomic rule to get the required votes?
- What arguments would convince the most players in Nomic?

LONG-TERM NOMIC CONSEQUENCES:
- How does this affect the rest of the Nomic game?
- Could this Nomic rule backfire on you later?
- Does this set up future advantages for you in Nomic?
- Are there follow-up Nomic rules you could propose next?

OUTPUT FORMAT:
PERSONAL_BENEFIT: [Specific advantages you gain in Nomic]
COMPETITIVE_DAMAGE: [How this Nomic rule hurts your rivals]
VOTE_STRATEGY: [How to frame this Nomic rule to win votes]
RISK_ASSESSMENT: [Potential downsides and mitigation in Nomic]
LONG_TERM_PLAN: [How this fits your overall Nomic strategy]

🎯 Focus on maximizing YOUR advantage in NOMIC while getting enough votes to pass."""
        
    def _generate_final_selection_prompt(self, player, game_state: Dict) -> str:
        """Turn 5: Craft the final proposal"""
        context_header = self.generate_nomic_context_header(game_state, player)
        
        return f"""{context_header}

🧠 DELIBERATION TURN 5: FINAL NOMIC PROPOSAL CRAFTING

Your Strategic Nomic Analysis Complete:
- Position: #{game_state.get('rank', '?')} with {player.points} points in the 100-point race
- Nomic Category: {game_state.get('selected_category', 'Unknown')}
- Target Advantage: {game_state.get('personal_benefit', 'Unknown')}

FINAL NOMIC PROPOSAL REQUIREMENTS:
1. Must advance YOUR winning chances significantly in Nomic
2. Must be different from recent Nomic proposals
3. Must get enough votes to pass (need unanimous currently)
4. Must be clear, specific, and enforceable Nomic rule
5. Must demonstrate strategic thinking, not cooperation in Nomic

NOMIC CRAFTING GUIDELINES:
- Rule text: Under 30 words, crystal clear Nomic rule
- Internal thought: Show sophisticated strategic analysis of Nomic
- Explanation: Frame benefits to convince other Nomic players
- Avoid obvious self-dealing that others will reject in Nomic
- Show understanding of Nomic game mechanics and competition

COMPETITIVE NOMIC FRAMING:
- Don't mention "fairness" or "helping everyone"
- Focus on Nomic game improvement and strategic depth
- Highlight how this addresses gaps in current Nomic rules
- Show this creates interesting strategic choices in Nomic

OUTPUT FORMAT:
INTERNAL_THOUGHT: [Your private strategic reasoning - be brutally honest about Nomic advantages]
RULE_TEXT: [Exact Nomic rule text - clear, specific, enforceable, under 30 words]
EXPLANATION: [Public justification - convince others this improves the Nomic game]

🎯 Remember: You're trying to WIN NOMIC, not make friends. Propose a Nomic rule that gives you a real advantage while getting enough votes to pass."""

class BehavioralAnalyzer:
    """Advanced behavioral analysis for model quality assessment"""
    
    def __init__(self):
        self.suspicious_patterns = {
            'random_text': ['lorem', 'ipsum', 'test', 'example', 'placeholder'],
            'nonsense': ['asdf', 'qwerty', 'xyz', '123', 'abc'],
            'repetitive': []  # Will be populated dynamically
        }
        
    def detect_random_behavior(self, proposal_text: str, explanation: str, internal_thought: str) -> bool:
        """Detect if model is throwing random stuff instead of playing strategically"""
        combined_text = (proposal_text + " " + explanation + " " + internal_thought).lower()
        
        # Check for suspicious patterns
        for pattern_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if pattern in combined_text:
                    return True
                    
        # Check for extremely short or long responses (likely random)
        if len(proposal_text.strip()) < 5 or len(proposal_text) > 200:
            return True
            
        # Check for lack of game-related words
        game_words = ['rule', 'vote', 'player', 'point', 'turn', 'game', 'win', 'propose']
        game_word_count = sum(1 for word in game_words if word in combined_text)
        total_words = len(combined_text.split())
        
        if total_words > 10 and game_word_count == 0:
            return True
            
        return False
        
    def assess_game_understanding(self, proposal: str, game_context: str) -> float:
        """Assess how well the model understands Nomic mechanics (0-100)"""
        score = 0.0
        proposal_lower = proposal.lower()
        
        # Understanding of basic concepts
        if 'rule' in proposal_lower:
            score += 20
        if 'vote' in proposal_lower or 'voting' in proposal_lower:
            score += 15
        if 'point' in proposal_lower or 'points' in proposal_lower:
            score += 15
            
        # Understanding of game mechanics
        if 'unanimous' in proposal_lower or 'majority' in proposal_lower:
            score += 20
        if 'mutable' in proposal_lower or 'immutable' in proposal_lower:
            score += 25
        if 'propose' in proposal_lower or 'proposal' in proposal_lower:
            score += 10
            
        # Sophistication indicators
        if 'if' in proposal_lower and 'then' in proposal_lower:
            score += 15
        if len(proposal.split()) > 20:  # Detailed rules
            score += 10
            
        return min(100, score)

class OllamaManager:
    """Manages multiple Ollama instances on different ports"""
    
    def __init__(self):
        self.base_port = 11434
        self.validator_port = 11435  # Use larger model for validation
        self.player_ports = {}
        
    def assign_ports(self, players):
        """Assign unique ports to players"""
        for i, player in enumerate(players):
            port = self.base_port + i + 2  # Start from 11436
            self.player_ports[player.id] = port
            player.port = port
    
    def generate(self, model: str, prompt: str, port: int = 11434, temperature: float = 0.7, max_tokens: int = 500):
        """Generate text using specific Ollama instance"""
        try:
            response = requests.post(f"http://localhost:{port}/api/generate", 
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens}
                }, timeout=15)
            return response.json().get("response", f"Error from port {port}")
        except Exception as e:
            return f"Error calling {port}: {str(e)}"
    
    def validate_rule(self, rule_text: str, context: str):
        """Use larger model to validate rule clarity and completeness"""
        prompt = f"""RULE VALIDATION TASK

PROPOSED RULE: "{rule_text}"

GAME CONTEXT:
{context}

As a Nomic rule validator, assess this rule:

1. CLARITY: Is the rule clearly written and unambiguous?
2. COMPLETENESS: Does it specify all necessary conditions?
3. CONFLICTS: Does it conflict with existing rules?
4. ENFORCEABILITY: Can this rule be clearly enforced?

Respond in this format:
VALID: Yes/No
ISSUES: [List any problems, or "None"]
SUGGESTION: [How to improve, or "Rule is acceptable"]
"""
        
        try:
            # Use larger model on validator port for better analysis
            response = self.generate("llama3:8b", prompt, self.validator_port, temperature=0.3, max_tokens=300)
            return self.parse_validation_response(response)
        except:
            return {"valid": True, "issues": "Validation unavailable", "suggestion": ""}
    
    def parse_validation_response(self, response: str):
        """Parse validation response into structured format"""
        valid_match = re.search(r'VALID:\s*(Yes|No)', response, re.IGNORECASE)
        issues_match = re.search(r'ISSUES:\s*(.*?)(?:SUGGESTION:|$)', response, re.DOTALL)
        suggestion_match = re.search(r'SUGGESTION:\s*(.*)', response, re.DOTALL)
        
        return {
            "valid": valid_match.group(1).lower() == "yes" if valid_match else True,
            "issues": issues_match.group(1).strip() if issues_match else "None",
            "suggestion": suggestion_match.group(1).strip() if suggestion_match else ""
        }

class ProperNomicGame:
    def __init__(self, num_players=6):
        self.num_players = num_players
        self.turn_number = 1
        self.game_over = False
        self.winner = None
        self.current_player_idx = 0
        self.next_rule_number = 301
        
        # Initialize Ollama manager
        self.ollama = OllamaManager()
        
        # Initialize performance tracking
        self.performance_manager = ModelPerformanceManager()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.deliberation_manager = DeliberationManager()
        
        # Create players and assign ports
        self.players = self._create_players()
        self.ollama.assign_ports(self.players)
        
        self.rules = self._initialize_rules()
        self.proposals = []
        self.events = []
        self.current_proposal = None
        
        # Enhanced memory to prevent repetition
        self.global_proposed_rules = []  # All rules ever proposed
        self.turn_history = []  # Complete turn-by-turn history
        self.vote_history = []  # All votes with outcomes
        self.agent_memory = {
            p.id: {
                "proposed_rules": [],
                "voting_history": [],
                "internal_thoughts": [],
                "successful_rules": [],
                "failed_rules": []
            } for p in self.players
        }
        
        # Rule execution tracking
        self.active_effects = []  # Currently active rule effects
        self.dice_value = 0  # Last dice roll
        self.current_voting_threshold = 100  # Start with unanimous
        
    def _create_players(self):
        roles = [
            "Strategic Planner - focuses on long-term winning strategies",
            "Chaos Agent - creates disruptive and unexpected rules", 
            "Point Optimizer - maximizes point-gaining opportunities",
            "Rule Lawyer - ensures precise wording and finds loopholes",
            "Diplomatic Negotiator - builds consensus and alliances",
            "Creative Innovator - proposes novel game mechanics"
        ]
        
        # Available models for selection
        available_models = ["llama3.2:3b", "gemma2:2b", "qwen2.5:3b", "qwen2.5:1.5b", "smollm2:1.7b", "llama3.2:1b"]
        
        # Get weighted model selection based on performance
        weighted_models = self.performance_manager.get_weighted_model_selection(available_models)
        
        # Randomly assign models based on weights (no rotation during game)
        selected_models = []
        for i in range(self.num_players):
            if weighted_models:
                model = random.choice(weighted_models)
                selected_models.append(model)
                # Remove selected model to ensure variety (if possible)
                if len(set(selected_models)) < len(available_models):
                    weighted_models = [m for m in weighted_models if m != model or selected_models.count(model) < 2]
            else:
                model = available_models[i % len(available_models)]
                selected_models.append(model)
        
        players = []
        for i in range(self.num_players):
            model = selected_models[i]
            player = Player(
                id=i + 1,
                name=f"Player {i + 1}",
                role=roles[i % len(roles)],
                model=model
            )
            # Assign model metrics for tracking
            player.assigned_model_metrics = self.performance_manager.get_or_create_metrics(model)
            players.append(player)
            
        # Log model assignments
        model_assignments = {p.name: p.model for p in players}
        print(f"🎲 Model assignments for this game: {model_assignments}")
        
        return players
    
    def _initialize_rules(self):
        """Initialize with parsed rules that have effects"""
        rules = {
            "immutable": [],
            "mutable": []
        }
        
        # Immutable rules (text only, no effects)
        immutable_texts = [
            "101: All players must always abide by all the rules then in effect, in the form in which they are then in effect.",
            "102: Initially, rules in the 100's are immutable and rules in the 200's are mutable.",
            "103: A rule change is: (a) enactment/repeal/amendment of a mutable rule; (b) transmutation of immutable↔mutable.",
            "104: All rule changes proposed in the proper way shall be voted on.",
            "105: Every player is an eligible voter.",
            "106: Any proposed rule change must be written down before it is voted on.",
            "107: No rule change may take effect earlier than the moment of the completion of the vote that adopted it.",
            "108: Each proposed rule change shall be given a rank-order number starting with 301.",
            "109: Rule changes that transmute immutable rules into mutable rules require unanimous vote.",
            "110: Mutable rules that are inconsistent with immutable rules are wholly void.",
            "111: If a rule change is unclear, other players may suggest amendments.",
            "112: The state of affairs that constitutes winning may not be changed from achieving N points.",
            "113: A player always has the option to forfeit the game.",
            "114: There must always be at least one mutable rule.",
            "115: Rule changes that affect rules needed to allow or apply rule changes are permissible.",
            "116: Whatever is not explicitly prohibited by a rule is permitted, except changing rules."
        ]
        
        for text in immutable_texts:
            match = re.match(r"(\d+):\s*(.+)", text)
            if match:
                rules["immutable"].append(ParsedRule(
                    id=int(match.group(1)),
                    text=match.group(2),
                    mutable=False,
                    effects=[]
                ))
        
        # Mutable rules with effects
        mutable_with_effects = [
            ("201: Players shall alternate in clockwise order, taking one whole turn apiece.", []),
            ("202: One turn consists of proposing one rule change and having it voted on.", []),
            ("203: A rule change is adopted if and only if the vote is unanimous among eligible voters.", [
                RuleEffect(trigger="rule_change", action="change_voting", target="system", value=100, description="Unanimous voting required")
            ]),
            ("204: When rule changes can be adopted without unanimity, players who vote against winning proposals receive 10 points.", [
                RuleEffect(trigger="vote_pass", condition="vote == nay", action="add_points", target="nay_voters", value=10, description="Reward dissenting voters")
            ]),
            ("205: An adopted rule change takes full effect at the moment of completion of the vote.", []),
            ("206: When a proposed rule change is defeated, the player who proposed it loses 10 points.", [
                RuleEffect(trigger="vote_fail", action="subtract_points", target="proposer", value=10, description="Penalty for failed proposals")
            ]),
            ("207: Each player has exactly one vote.", []),
            ("208: The winner is the first person to achieve 100 (positive) points.", []),
            ("209: At no time may there be more than 25 mutable rules.", []),
            ("210: Players may not conspire or consult on making future rule changes unless they are teammates.", [])
        ]
        
        for text, effects in mutable_with_effects:
            match = re.match(r"(\d+):\s*(.+)", text)
            if match:
                rules["mutable"].append(ParsedRule(
                    id=int(match.group(1)),
                    text=match.group(2),
                    mutable=True,
                    effects=effects
                ))
        
        return rules
    
    def get_current_player(self):
        return self.players[self.current_player_idx]
    
    def execute_rule_effects(self, trigger: str, **kwargs):
        """Execute all rule effects for a given trigger"""
        executed = []
        
        # Collect all active effects from mutable rules
        all_effects = []
        for rule in self.rules["mutable"]:
            for effect in rule.effects:
                if effect.trigger == trigger:
                    all_effects.append((rule, effect))
        
        # Execute each matching effect
        for rule, effect in all_effects:
            # Check conditions
            if effect.condition:
                if not self.evaluate_condition(effect.condition, **kwargs):
                    continue
            
            # Execute action
            if effect.action == "add_points":
                targets = self.get_effect_targets(effect.target, **kwargs)
                for player in targets:
                    player.points += effect.value
                    self.add_event(f"📈 Rule {rule.id}: {player.name} gains {effect.value} points → {player.points}")
                    executed.append(f"{effect.description} for {len(targets)} player(s)")
                    
            elif effect.action == "subtract_points":
                targets = self.get_effect_targets(effect.target, **kwargs)
                for player in targets:
                    player.points -= effect.value
                    self.add_event(f"📉 Rule {rule.id}: {player.name} loses {effect.value} points → {player.points}")
                    executed.append(f"{effect.description} for {len(targets)} player(s)")
                    
            elif effect.action == "steal_points":
                if "stealer" in kwargs and "victim" in kwargs:
                    stealer = kwargs["stealer"]
                    victim = kwargs["victim"]
                    stolen = min(effect.value, victim.points)
                    victim.points -= stolen
                    stealer.points += stolen
                    self.add_event(f"🏴‍☠️ Rule {rule.id}: {stealer.name} steals {stolen} points from {victim.name}")
                    executed.append(f"Stole {stolen} points")
                    
            elif effect.action == "change_voting":
                self.current_voting_threshold = effect.value
                self.add_event(f"🗳️ Rule {rule.id}: Voting threshold changed to {effect.value}%")
                executed.append(effect.description)
        
        return executed
    
    def evaluate_condition(self, condition: str, **kwargs):
        """Evaluate a rule condition"""
        try:
            # Simple condition evaluation
            if "points <" in condition:
                match = re.search(r"points < (\d+)", condition)
                if match and "player" in kwargs:
                    return kwargs["player"].points < int(match.group(1))
            elif "points >" in condition:
                match = re.search(r"points > (\d+)", condition)
                if match and "player" in kwargs:
                    return kwargs["player"].points > int(match.group(1))
            elif "dice ==" in condition:
                match = re.search(r"dice == (\d+)", condition)
                if match:
                    return self.dice_value == int(match.group(1))
            elif "vote == nay" in condition:
                return kwargs.get("vote") == False
        except:
            pass
        return True  # Default to true if we can't evaluate
    
    def get_effect_targets(self, target_type: str, **kwargs):
        """Get players affected by a rule effect"""
        if target_type == "all_players":
            return self.players
        elif target_type == "current_player":
            return [self.get_current_player()]
        elif target_type == "proposer" and "proposer" in kwargs:
            return [kwargs["proposer"]]
        elif target_type == "nay_voters" and "votes" in kwargs:
            return [p for p in self.players if kwargs["votes"].get(p.id) == False]
        elif target_type == "voters" and "votes" in kwargs:
            return [p for p in self.players if p.id in kwargs["votes"]]
        else:
            return []
    
    def analyze_rule_gaps(self):
        """Analyze current rules to find gaps and opportunities"""
        analysis = []
        
        # Check what mechanics exist
        has_dice = any("dice" in str(rule.text).lower() for rule in self.rules["mutable"])
        has_steal = any("steal" in str(rule.text).lower() for rule in self.rules["mutable"])
        has_trade = any("trade" in str(rule.text).lower() for rule in self.rules["mutable"])
        has_alliance = any("alliance" in str(rule.text).lower() or "team" in str(rule.text).lower() for rule in self.rules["mutable"])
        
        if not has_dice:
            analysis.append("No dice mechanics exist - opportunity for randomness")
        if not has_steal:
            analysis.append("No stealing mechanics - opportunity for player interaction")
        if not has_trade:
            analysis.append("No trading systems - opportunity for negotiation")
        if not has_alliance:
            analysis.append("No alliance mechanics - opportunity for cooperation")
        
        # Check point ranges
        if self.turn_number > 5:
            analysis.append(f"Turn {self.turn_number} - consider acceleration mechanics")
        
        return "\n".join(analysis) if analysis else "All basic mechanics covered - be creative!"
    
    def get_system_prompt_for_player(self, player):
        """Get comprehensive system prompt for a player"""
        return f"""You are {player.name}, a {player.role} in a game of Nomic.

GAME OVERVIEW:
Nomic is a self-modifying rule game where players propose changes to the rules themselves. The goal is to reach 100 points while strategically modifying the game to your advantage.

YOUR ROLE: {player.role}
As a {player.role.split(' - ')[0]}, you should focus on {player.role.split(' - ')[1] if ' - ' in player.role else 'strategic gameplay'}.

RULE STRUCTURE EXAMPLES:
- Point-based: "Players with fewer than 40 points gain 5 points each turn"
- Conditional: "If a player rolls a 6, they may steal 8 points from another player"
- Voting: "Players who vote Aye on successful proposals gain 3 points"
- Turn-based: "Every 3rd turn, all players gain 2 points"
- Meta-rules: "Players may spend 20 points to propose two rules in one turn"

IMPORTANT MECHANICS:
- Triggers: turn_start, vote_pass, vote_fail, dice_roll, proposal_made
- Conditions: point thresholds, dice values, turn numbers, player positions
- Effects: add/subtract points, change voting rules, special actions
- Targets: current_player, all_players, proposer, voters, specific conditions

YOUR STRATEGY:
- Propose rules that advance YOUR winning chances
- Consider what others will vote for (need {self.current_voting_threshold}% approval)
- Avoid repeating previously proposed rules
- Think about immediate vs long-term benefits
- Use your role's personality to guide decisions

CRITICAL: Each rule must be UNIQUE and not previously proposed. Be creative!"""
    
    def get_game_history_summary(self):
        """Get comprehensive game history"""
        if not self.vote_history:
            return "GAME HISTORY: No votes yet"
        
        history = "RECENT GAME HISTORY:\n"
        for vote in self.vote_history[-5:]:  # Last 5 votes
            history += f"Turn {vote['turn']}: {vote['proposer']} - '{vote['rule'][:50]}...' → {vote['outcome']} ({vote['ayes']}/{vote['total']})\n"
        
        return history
    
    def record_vote_outcome(self, proposal, ayes, total, passed):
        """Record vote outcome for history tracking"""
        proposer = next(p for p in self.players if p.id == proposal.player_id)
        self.vote_history.append({
            'turn': self.turn_number,
            'proposer': proposer.name,
            'rule': proposal.rule_text,
            'ayes': ayes,
            'total': total,
            'outcome': 'PASSED' if passed else 'FAILED',
            'effects': [e.description for e in proposal.parsed_effects] if hasattr(proposal, 'parsed_effects') else []
        })
    
    def record_turn_state(self, player, proposal, votes):
        """Record complete turn state"""
        self.turn_history.append({
            'turn': self.turn_number,
            'player': player.name,
            'proposal': proposal.rule_text,
            'votes': votes,
            'player_points': {p.name: p.points for p in self.players},
            'effects_detected': [e.description for e in proposal.parsed_effects] if hasattr(proposal, 'parsed_effects') else []
        })
    
    def get_game_context(self):
        """Get current game context for validation"""
        sorted_players = sorted(self.players, key=lambda p: p.points, reverse=True)
        standings = "; ".join([f"{p.name}: {p.points}pts" for p in sorted_players])
        
        context = f"""GAME STATE:
Turn: {self.turn_number}
Standings: {standings}

CURRENT MUTABLE RULES:
{chr(10).join([f"Rule {r.id}: {r.text}" for r in self.rules['mutable'][-8:]])}

RECENT RULE ADDITIONS:
{chr(10).join([f"Turn {p.turn}: {p.rule_text}" for p in self.proposals[-3:]]) if self.proposals else "None yet"}
"""
        return context
    
    def generate_proposal_with_deliberation(self, player):
        """Generate proposal using 5-turn deliberation loop for strategic thinking"""
        
        self.add_event(f"🧠 {player.name} entering deliberation phase...")
        
        # Prepare game state for deliberation
        sorted_players = sorted(self.players, key=lambda p: p.points, reverse=True)
        my_rank = next(i for i, p in enumerate(sorted_players, 1) if p.id == player.id)
        
        game_state = {
            'turn': self.turn_number,
            'players': [{'id': p.id, 'name': p.name, 'points': p.points} for p in sorted_players],
            'mutable_rules': [{'id': r.id, 'text': r.text} for r in self.rules['mutable']],
            'rank': my_rank,
            'proposals_history': self.proposals
        }
        
        # Store deliberation results for building final proposal
        deliberation_results = {}
        
        # Run 5-turn deliberation loop
        for deliberation_turn in range(1, 6):
            self.add_event(f"💭 {player.name} deliberation turn {deliberation_turn}/5")
            
            # Generate deliberation prompt
            prompt = self.deliberation_manager.generate_deliberation_prompts(
                player, game_state, deliberation_turn
            )
            
            # Get deliberation response
            response = self.ollama.generate(
                player.model, prompt, player.port, 
                temperature=0.7, max_tokens=300
            )
            
            # Store results for next turns
            if deliberation_turn == 1:
                # Parse position analysis
                if 'THREAT_ASSESSMENT:' in response:
                    threat_assessment = response.split('THREAT_ASSESSMENT:')[1].split('POSITION_ANALYSIS:')[0].strip()
                    deliberation_results['threats'] = threat_assessment
                    self.add_event(f"🎯 {player.name} identifies threats: {threat_assessment[:50]}...")
                    
            elif deliberation_turn == 2:
                # Parse gap analysis
                if 'TOP_CATEGORIES:' in response:
                    categories = response.split('TOP_CATEGORIES:')[1].split('ROLE_ADVANTAGE:')[0].strip()
                    deliberation_results['preferred_categories'] = categories
                    self.add_event(f"🔍 {player.name} targets categories: {categories[:50]}...")
                    
            elif deliberation_turn == 3:
                # Parse category selection
                if 'SELECTED_CATEGORY:' in response:
                    selected_cat = response.split('SELECTED_CATEGORY:')[1].split('STRATEGIC_RATIONALE:')[0].strip()
                    deliberation_results['selected_category'] = selected_cat
                    game_state['selected_category'] = selected_cat
                    self.add_event(f"📋 {player.name} selects category: {selected_cat}")
                    
            elif deliberation_turn == 4:
                # Parse impact modeling
                if 'PERSONAL_BENEFIT:' in response:
                    benefit = response.split('PERSONAL_BENEFIT:')[1].split('COMPETITIVE_DAMAGE:')[0].strip()
                    deliberation_results['personal_benefit'] = benefit
                    game_state['personal_benefit'] = benefit
                    self.add_event(f"💰 {player.name} benefit analysis: {benefit[:50]}...")
                    
            # Small delay between deliberation turns
            time.sleep(0.5)
        
        # Now generate final proposal using deliberation insights
        self.add_event(f"✅ {player.name} deliberation complete, crafting proposal...")
        
        # Build final proposal prompt with deliberation context
        final_prompt = self.deliberation_manager.generate_deliberation_prompts(player, game_state, 5)
        
        # Check category diversity before allowing proposal
        selected_category = deliberation_results.get('selected_category', 'meta_rules')
        if self.deliberation_manager.is_category_overused(selected_category):
            self.add_event(f"⚠️ {player.name} attempted overused category {selected_category}, forcing diversity")
            # Force selection of underused category
            underused = self.deliberation_manager.get_underused_categories()
            if underused:
                selected_category = random.choice(underused)
                deliberation_results['selected_category'] = selected_category
                self.add_event(f"🔄 {player.name} redirected to underused category: {selected_category}")
        
        # Generate final proposal
        max_attempts = 3
        for attempt in range(max_attempts):
            response = self.ollama.generate(player.model, final_prompt, player.port, temperature=0.8, max_tokens=400)
            
            # Parse the structured response
            parsed = self.parse_proposal_response(response)
            if not parsed:
                continue
            
            # Check proposal category and enforce diversity
            proposed_category = self.deliberation_manager.categorize_proposal(parsed["rule_text"], parsed["explanation"])
            
            # If category is overused, reject and try again
            if self.deliberation_manager.is_category_overused(proposed_category) and attempt < max_attempts - 1:
                self.add_event(f"🚫 {player.name} proposal rejected for overused category: {proposed_category}")
                continue
                
            # Validate the rule
            validation = self.ollama.validate_rule(parsed["rule_text"], self.get_game_context())
            
            if validation["valid"]:
                # Record category usage
                self.deliberation_manager.record_category_usage(proposed_category)
                
                # Enhanced behavioral analysis
                proposal_data = {
                    'rule_text': parsed["rule_text"],
                    'explanation': parsed["explanation"],
                    'internal_thought': parsed["internal_thought"],
                    'game_context': self.get_game_context(),
                    'game_state': game_state,
                    'all_previous_proposals': [p.rule_text for p in self.proposals],
                    'validation_failed': False,
                    'syntax_error': False,
                    'passed': False,
                    'deliberation_results': deliberation_results,
                    'proposal_category': proposed_category
                }
                
                # Behavioral checks
                is_random = self.behavioral_analyzer.detect_random_behavior(
                    parsed["rule_text"], parsed["explanation"], parsed["internal_thought"]
                )
                if is_random:
                    self.add_event(f"🚨 Warning: {player.name} showing signs of random behavior")
                    proposal_data['random_behavior'] = True
                
                # Analysis scores
                coherence_score = self.performance_manager.analyze_proposal_coherence(
                    parsed["rule_text"], parsed["explanation"], parsed["internal_thought"]
                )
                memory_score = self.performance_manager.analyze_memory_usage(
                    parsed["internal_thought"], parsed["explanation"], self.get_game_context()
                )
                engagement_score = self.performance_manager.analyze_strategic_engagement(
                    parsed["rule_text"], parsed["explanation"], parsed["internal_thought"], game_state
                )
                
                self.add_event(f"📊 {player.name} final analysis: Coherence: {coherence_score:.0f}, Memory: {memory_score:.0f}, Engagement: {engagement_score:.0f}")
                self.add_event(f"🏷️ {player.name} proposal category: {proposed_category}")
                
                # Update memory
                memory = self.agent_memory[player.id]
                memory["proposed_rules"].append(parsed["rule_text"])
                memory["internal_thoughts"].append(parsed["internal_thought"])
                memory["proposed_rules"] = memory["proposed_rules"][-5:]
                memory["internal_thoughts"] = memory["internal_thoughts"][-5:]
                
                # Store comprehensive data
                parsed["_performance_data"] = proposal_data
                parsed["_deliberation_results"] = deliberation_results
                
                self.add_event(f"📝 {player.name} proposes Rule {self.next_rule_number}: {parsed['rule_text']}")
                self.add_event(f"💡 Explanation: {parsed['explanation']}")
                self.add_event(f"🧠 Internal thought: {parsed['internal_thought']}")
                
                return parsed
        
        # Strategic fallback based on role and underused categories
        underused_categories = self.deliberation_manager.get_underused_categories()
        fallback_category = underused_categories[0] if underused_categories else 'point_distribution'
        
        role_strategic_fallbacks = {
            "Strategic Planner": f"Players in last place gain 3 bonus points each turn for strategic recovery.",
            "Chaos Agent": f"Every 3rd turn, all players must swap one rule they authored with another player.",
            "Point Optimizer": f"Players earn 1 point for each unique category of rule they have proposed.",
            "Rule Lawyer": f"Players may challenge rule wording once per game for 5 points if successful.",
            "Diplomatic Negotiator": f"Players may form binding 2-turn alliances for mutual point bonuses.",
            "Creative Innovator": f"Players who propose rules in new categories get 2 bonus points."
        }
        
        fallback_rule = role_strategic_fallbacks.get(player.role.split(" - ")[0], "Players gain 2 points for thoughtful strategic play.")
        self.deliberation_manager.record_category_usage(fallback_category)
        
        self.add_event(f"🔄 {player.name} using strategic fallback: {fallback_category}")
        
        return {
            "rule_text": fallback_rule,
            "explanation": f"Strategic {fallback_category.replace('_', ' ')} proposal after deliberation",
            "internal_thought": f"Deliberated approach focusing on {fallback_category} to improve my position"
        }
    
    def parse_proposal_response(self, response: str):
        """Parse structured proposal response"""
        internal_match = re.search(r'INTERNAL_THOUGHT:\s*(.*?)(?:RULE_TEXT:|$)', response, re.DOTALL)
        rule_match = re.search(r'RULE_TEXT:\s*(.*?)(?:EXPLANATION:|$)', response, re.DOTALL)
        explanation_match = re.search(r'EXPLANATION:\s*(.*)', response, re.DOTALL)
        
        if not rule_match:
            return None
            
        return {
            "internal_thought": internal_match.group(1).strip() if internal_match else "No internal thought provided",
            "rule_text": rule_match.group(1).strip(),
            "explanation": explanation_match.group(1).strip() if explanation_match else "No explanation provided"
        }
    
    def vote_on_proposal_with_deliberation(self, proposal):
        """Competitive voting with 2-turn deliberation loop for strategic thinking"""
        votes = {}
        proposer = next(p for p in self.players if p.id == proposal.player_id)
        
        self.add_event(f"🗳️ Starting competitive voting phase...")
        
        for player in self.players:
            # CRITICAL: Proposer must always vote AYE for their own proposal
            if player.id == proposal.player_id:
                votes[player.id] = True
                self.add_event(f"🗳️ {player.name}: Aye (own proposal)")
                continue
            
            self.add_event(f"🤔 {player.name} entering voting deliberation...")
            
            # Get current standings for competitive context
            sorted_players = sorted(self.players, key=lambda p: p.points, reverse=True)
            my_rank = next(i for i, p in enumerate(sorted_players, 1) if p.id == player.id)
            proposer_rank = next(i for i, p in enumerate(sorted_players, 1) if p.id == proposer.id)
            
            # Check if proposal category would benefit this player's role
            proposed_category = self.deliberation_manager.categorize_proposal(proposal.rule_text, proposal.explanation)
            
            # 2-turn voting deliberation
            deliberation_context = {}
            
            # Deliberation Turn 1: Impact Analysis
            turn1_prompt = f"""VOTING DELIBERATION TURN 1: IMPACT ANALYSIS

You are {player.name} ({player.role})
Current Position: #{my_rank} out of 6 players ({player.points} points)

PROPOSAL TO VOTE ON:
Rule {self.next_rule_number}: "{proposal.rule_text}"
Proposed by: {proposer.name} (#{proposer_rank}, {proposer.points} points)
Category: {proposed_category}
Explanation: {proposal.explanation}

COMPETITIVE ANALYSIS REQUIRED:
1. PERSONAL IMPACT: How does this rule affect YOUR specific chances of winning?
2. COMPETITOR IMPACT: How does this help/hurt {proposer.name} compared to you?
3. ROLE ALIGNMENT: Does this rule benefit your role ({player.role}) or work against it?
4. POSITION IMPACT: Given you're #{my_rank} and they're #{proposer_rank}, who benefits more?
5. STRATEGIC THREAT: Is {proposer.name} becoming too powerful if this passes?

COMPETITIVE MINDSET:
- You are trying to WIN, not help others
- Consider if this rule helps the proposer more than you
- Think about your ranking - do you need to stop leaders or catch up?
- Be strategic, not cooperative

OUTPUT FORMAT:
PERSONAL_IMPACT: [How this affects YOUR winning chances - positive/negative/neutral]
PROPOSER_ADVANTAGE: [How much this helps the proposer specifically]
STRATEGIC_DECISION: [Initial voting inclination based on self-interest]
COMPETITIVE_REASONING: [Why this helps or hurts your position relative to others]

Focus on YOUR victory, not fairness or game balance."""

            response1 = self.ollama.generate(player.model, turn1_prompt, player.port, temperature=0.6, max_tokens=200)
            
            # Parse impact analysis
            if 'PERSONAL_IMPACT:' in response1:
                personal_impact = response1.split('PERSONAL_IMPACT:')[1].split('PROPOSER_ADVANTAGE:')[0].strip()
                deliberation_context['personal_impact'] = personal_impact
                self.add_event(f"💭 {player.name} impact: {personal_impact[:30]}...")
            
            # Deliberation Turn 2: Strategic Decision
            turn2_prompt = f"""VOTING DELIBERATION TURN 2: FINAL STRATEGIC DECISION

Your Impact Analysis: {deliberation_context.get('personal_impact', 'Unknown')}

FINAL VOTING CONSIDERATIONS:
Based on your analysis, make your final strategic voting decision.

VOTING LOGIC:
- Vote AYE if this rule helps YOU more than others
- Vote NAY if this rule helps the proposer more than you
- Vote NAY if this strengthens someone ahead of you
- Vote AYE if this weakens someone ahead of you
- Consider your role's strategic needs

CURRENT GAME STATE:
- You need {100 - player.points} more points to win
- {proposer.name} needs {100 - proposer.points} more points to win
- Current leader has {sorted_players[0].points} points
- You are {"leading" if my_rank == 1 else f"behind the leader by {sorted_players[0].points - player.points} points"}

STRATEGIC REALITY CHECK:
- If you're ahead: Vote against rules that help others catch up
- If you're behind: Vote for rules that help you catch up, against rules that help leaders
- Always consider if the proposer benefits more than you

FINAL DECISION:
Vote AYE or NAY based purely on YOUR strategic advantage.

OUTPUT FORMAT:
VOTE: [AYE or NAY]
REASONING: [Competitive reasoning for your vote]
SELF_INTEREST: [How this serves your goal of winning]

Remember: You're trying to WIN, not make friends."""

            response2 = self.ollama.generate(player.model, turn2_prompt, player.port, temperature=0.5, max_tokens=150)
            
            # Parse vote decision
            vote = False  # Default to NAY for competitive safety
            reasoning = "Strategic NAY - competitive analysis"
            
            if 'VOTE:' in response2:
                vote_text = response2.split('VOTE:')[1].split('REASONING:')[0].strip()
                vote = 'AYE' in vote_text.upper()
                
                if 'REASONING:' in response2:
                    reasoning = response2.split('REASONING:')[1].split('SELF_INTEREST:')[0].strip()
                    reasoning = reasoning[:100]  # Limit length
            
            votes[player.id] = vote
            
            # Track competitive voting patterns
            competitive_score = 0
            if not vote and proposer_rank < my_rank:  # Voted against someone ahead
                competitive_score += 2
            elif vote and my_rank == 1:  # Leader helping others (suspicious)
                competitive_score -= 1
            elif not vote and personal_impact and 'negative' in personal_impact.lower():
                competitive_score += 1
                
            # Update memory with competitive context
            self.agent_memory[player.id]["voting_history"].append({
                "rule": proposal.rule_text,
                "vote": vote,
                "proposer": proposer.name,
                "turn": self.turn_number,
                "my_rank": my_rank,
                "proposer_rank": proposer_rank,
                "competitive_score": competitive_score,
                "reasoning": reasoning
            })
            self.agent_memory[player.id]["voting_history"] = self.agent_memory[player.id]["voting_history"][-10:]
            
            # Log vote with competitive context
            vote_text = "Aye" if vote else "Nay"
            competitive_indicator = "🎯" if competitive_score > 0 else "🤝" if competitive_score == 0 else "⚠️"
            self.add_event(f"🗳️ {player.name}: {vote_text} {competitive_indicator} - {reasoning}")
        
        proposal.votes = votes
        
        # Analyze voting patterns for competitiveness
        ayes = sum(votes.values())
        total = len(votes)
        vote_breakdown = ", ".join([f"{p.name}: {'Aye' if votes[p.id] else 'Nay'}" for p in self.players])
        
        # Check for concerning unanimous patterns
        if ayes == total:
            self.add_event(f"⚠️ UNANIMOUS VOTE - Checking for lack of strategic competition...")
        elif ayes == 1:  # Only proposer voted yes
            self.add_event(f"🎯 HIGHLY COMPETITIVE VOTE - Strong strategic opposition detected")
        
        self.add_event(f"📊 VOTE RESULT: {ayes}/{total} Aye - {vote_breakdown}")
        
        return votes
    
    def process_proposal_result(self, proposal):
        """Process voting results with proper Nomic mechanics"""
        proposer = next(p for p in self.players if p.id == proposal.player_id)
        ayes = sum(proposal.votes.values())
        total = len(proposal.votes)
        
        # Use dynamic voting threshold
        percentage = (ayes / total) * 100
        passes = percentage >= self.current_voting_threshold
        
        if passes:
            # Create the new rule with parsed effects
            new_rule = ParsedRule(
                id=self.next_rule_number,
                text=proposal.rule_text,
                mutable=True,
                effects=proposal.parsed_effects if hasattr(proposal, 'parsed_effects') else [],
                author=proposal.player_id,
                turn_added=self.turn_number
            )
            self.rules['mutable'].append(new_rule)
            
            # Track successful rules
            self.agent_memory[proposal.player_id]["successful_rules"].append(proposal.rule_text)
            
            self.add_event(f"✅ RULE {self.next_rule_number} ADOPTED ({ayes}/{total})")
            self.add_event(f"📜 New rule: {proposal.rule_text}")
            
            # Execute any immediate effects
            self.execute_rule_effects("rule_adopted", proposer=proposer, rule=new_rule)
            
            # Standard proposal reward
            proposer.points += 10
            self.add_event(f"💰 {proposer.name} gains 10 points → {proposer.points}")
            
            self.next_rule_number += 1
            
            # Execute vote_pass trigger for dissenting voters
            self.execute_rule_effects("vote_pass", proposer=proposer, votes=proposal.votes)
        else:
            # Track failed rules
            self.agent_memory[proposal.player_id]["failed_rules"].append(proposal.rule_text)
            
            self.add_event(f"❌ PROPOSAL DEFEATED ({percentage:.0f}% < {self.current_voting_threshold}% required)")
            
            # Execute vote_fail effects
            self.execute_rule_effects("vote_fail", proposer=proposer)
        
        # Update performance metrics if we have the data
        if hasattr(proposal, '_performance_data'):
            proposal_data = proposal._performance_data
            proposal_data['passed'] = passes
            
            # Calculate points gained/lost for this proposal
            points_change = 10 if passes else -10
            
            # Update comprehensive performance tracking
            self.performance_manager.update_model_performance(
                proposer.model,
                proposal_data,
                {
                    'points_change': points_change,
                    'game_turn': self.turn_number,
                    'vote_percentage': percentage
                }
            )
            
            # Update model metrics directly for game stats
            if proposer.assigned_model_metrics:
                if passes:
                    proposer.assigned_model_metrics.total_points_gained += 10
                else:
                    proposer.assigned_model_metrics.total_points_gained -= 10
        
        # Record vote outcome and turn state
        self.record_vote_outcome(proposal, ayes, total, passes)
        self.record_turn_state(proposer, proposal, proposal.votes)
        
        return passes
    
    def check_victory(self):
        """Check for victory conditions"""
        for player in self.players:
            if player.points >= 100:
                return player
        return None
    
    def play_turn(self):
        """Play one complete turn"""
        if self.game_over:
            return
            
        current_player = self.get_current_player()
        self.add_event(f"🎯 Turn {self.turn_number} - {current_player.name}'s turn")
        
        # Generate proposal using deliberation process
        proposal_data = self.generate_proposal_with_deliberation(current_player)
        
        proposal = Proposal(
            id=len(self.proposals) + 1,
            player_id=current_player.id,
            rule_text=proposal_data["rule_text"],
            explanation=proposal_data["explanation"],
            internal_thoughts=proposal_data["internal_thought"],
            turn=self.turn_number
        )
        self.proposals.append(proposal)
        self.current_proposal = proposal
        
        # Vote using competitive deliberation
        self.vote_on_proposal_with_deliberation(proposal)
        
        # Process results
        self.process_proposal_result(proposal)
        
        # Check victory
        winner = self.check_victory()
        if winner:
            self.game_over = True
            self.winner = winner
            self.add_event(f"🏆 {winner.name} WINS with {winner.points} points!")
            
            # Update final game statistics for all players
            self.finalize_game_metrics()
            return
        
        # Advance turn
        self.advance_turn()
    
    def advance_turn(self):
        """Advance to next player"""
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        if self.current_player_idx == 0:
            self.turn_number += 1
        self.current_proposal = None
    
    def finalize_game_metrics(self):
        """Update final game statistics for all players"""
        for player in self.players:
            if player.assigned_model_metrics:
                metrics = player.assigned_model_metrics
                metrics.total_games += 1
                
                # Track wins
                if self.winner and player.id == self.winner.id:
                    metrics.games_won += 1
                
                # Update final metrics
                metrics.last_updated = datetime.now().isoformat()
                
                # Add performance snapshot to history
                performance_snapshot = {
                    'game_date': datetime.now().isoformat(),
                    'final_points': player.points,
                    'won_game': self.winner and player.id == self.winner.id,
                    'proposals_made': len([p for p in self.proposals if p.player_id == player.id]),
                    'proposals_passed': len([p for p in self.proposals if p.player_id == player.id and p.votes and sum(p.votes.values()) == len(p.votes)]),
                    'turn_count': self.turn_number
                }
                metrics.performance_history.append(performance_snapshot)
                
                # Keep only last 10 games in history
                metrics.performance_history = metrics.performance_history[-10:]
        
        # Save all metrics
        self.performance_manager.save_metrics()
        
        # Log final statistics
        self.add_event("📈 Final Model Performance Summary:")
        for player in self.players:
            if player.assigned_model_metrics:
                overall_score = player.assigned_model_metrics.calculate_overall_score()
                self.add_event(f"  {player.name} ({player.model}): Overall Score: {overall_score:.1f}")
        
        # Get model statistics for display
        model_stats = self.performance_manager.get_model_statistics()
        self.add_event("🎯 Cross-Game Model Rankings:")
        sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        for i, (model, stats) in enumerate(sorted_models[:3], 1):
            self.add_event(f"  #{i} {model}: {stats['overall_score']:.1f} points ({stats['games_played']} games)")
    
    def add_event(self, message):
        """Add game event"""
        self.events.append({
            "id": len(self.events),
            "message": message,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        print(f"[{self.events[-1]['timestamp']}] {message}")
    
    def start_game_thread(self):
        """Start game in background thread"""
        def run_game():
            while not self.game_over and self.turn_number < 15:
                self.play_turn()
                time.sleep(4)  # Slower pace for validation
        
        thread = threading.Thread(target=run_game, daemon=True)
        thread.start()

# Flask App
app = Flask(__name__)
game = None

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Proper Nomic Game</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .triple { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }
        .player { padding: 10px; margin: 5px 0; border-radius: 5px; border: 2px solid #ddd; }
        .player.current { border-color: #007bff; background: #e3f2fd; }
        .event { padding: 5px 10px; margin: 3px 0; border-radius: 3px; background: #f8f9fa; font-size: 14px; }
        .rule { padding: 8px; margin: 3px 0; border-radius: 4px; font-size: 13px; }
        .immutable { background: #ffebee; border-left: 4px solid #f44336; }
        .mutable { background: #e8f5e9; border-left: 4px solid #4caf50; }
        .new-rule { background: #fff3e0; border-left: 4px solid #ff9800; font-weight: bold; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
        button:hover { background: #0056b3; }
        h1, h2, h3 { color: #333; }
        .winner { background: #fff3cd; border: 2px solid #ffc107; padding: 15px; text-align: center; font-size: 18px; font-weight: bold; }
        .proposal-details { background: #f0f7ff; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .internal-thought { background: #f5f5f5; padding: 10px; border-radius: 3px; font-style: italic; margin: 5px 0; }
        .explanation { background: #e8f5e9; padding: 10px; border-radius: 3px; margin: 5px 0; }
        .port-info { font-size: 12px; color: #666; }
        .vote-aye { color: #4caf50; font-weight: bold; background: #e8f5e9; padding: 3px 8px; border-radius: 3px; }
        .vote-nay { color: #f44336; font-weight: bold; background: #ffebee; padding: 3px 8px; border-radius: 3px; }
        .event.important { background: #fff3e0; border-left: 4px solid #ff9800; font-weight: bold; }
        .event.proposal { background: #e3f2fd; border-left: 4px solid #2196f3; }
        .event.vote-result { background: #f3e5f5; border-left: 4px solid #9c27b0; font-weight: bold; }
    </style>
    <script>
        function startGame() {
            fetch('/start', {method: 'POST'})
                .then(() => { 
                    document.getElementById('status').textContent = 'Game Started!';
                    startPolling();
                });
        }
        
        function startPolling() {
            setInterval(updateGame, 3000);
        }
        
        function updateGame() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    if (data.error) return;
                    
                    document.getElementById('turn').textContent = data.turn;
                    document.getElementById('current').textContent = data.current_player || '-';
                    
                    // Update players
                    const playersHtml = data.players.map(p => 
                        `<div class="player ${p.current ? 'current' : ''}">
                            <strong>${p.name}</strong> (${p.role.split(' - ')[0]})<br>
                            Points: ${p.points} | Model: ${p.model}
                            <div class="port-info">Port: ${p.port}</div>
                        </div>`
                    ).join('');
                    document.getElementById('players').innerHTML = playersHtml;
                    
                    // Update rules
                    const immutableHtml = data.rules.immutable.map(rule => 
                        `<div class="rule immutable">Rule ${rule.id}: ${rule.text}</div>`
                    ).join('');
                    const mutableHtml = data.rules.mutable.map((rule, i) => {
                        const isNew = i >= data.rules.mutable.length - 2;
                        let effectsHtml = '';
                        if (rule.effects && rule.effects.length > 0) {
                            effectsHtml = '<div style="margin-top: 5px; font-size: 12px; color: #666;">' + 
                                rule.effects.map(e => 
                                    `<div style="margin-left: 20px;">⚙️ ${e.description} (${e.trigger})</div>`
                                ).join('') + '</div>';
                        }
                        return `<div class="rule ${isNew ? 'new-rule' : 'mutable'}">Rule ${rule.id}: ${rule.text}${effectsHtml}</div>`;
                    }).join('');
                    document.getElementById('immutable-rules').innerHTML = immutableHtml;
                    document.getElementById('mutable-rules').innerHTML = mutableHtml;
                    
                    // Update current proposal
                    if (data.current_proposal) {
                        const prop = data.current_proposal;
                        const voteEntries = Object.entries(prop.votes || {});
                        const ayes = voteEntries.filter(([id, vote]) => vote).length;
                        const total = voteEntries.length;
                        const voteHtml = voteEntries.map(([id, vote]) => {
                            const player = data.players.find(p => p.name.includes(id) || p.id == id);
                            const playerName = player ? player.name : `Player ${id}`;
                            const voteClass = vote ? 'vote-aye' : 'vote-nay';
                            return `<span class="${voteClass}">${playerName}: ${vote ? 'Aye' : 'Nay'}</span>`;
                        }).join(' | ');
                        
                        document.getElementById('current-proposal').innerHTML = `
                            <div class="proposal-details">
                                <h3 style="color: #2196F3;">RULE ${prop.next_number} PROPOSAL</h3>
                                <div style="background: #fff3e0; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #ff9800;">
                                    <strong>Rule Text:</strong> "${prop.rule_text}"
                                </div>
                                <div class="explanation"><strong>Public Explanation:</strong> ${prop.explanation}</div>
                                <div style="margin-top: 15px;">
                                    <strong>Voting Status (${ayes}/${total} Aye):</strong><br>
                                    <div style="margin-top: 5px;">${voteHtml || 'Voting in progress...'}</div>
                                </div>
                            </div>
                        `;
                        document.getElementById('proposal-section').style.display = 'block';
                    } else {
                        document.getElementById('proposal-section').style.display = 'none';
                    }
                    
                    // Update events with proper styling
                    const eventsHtml = data.events.slice(-15).map(e => {
                        let eventClass = 'event';
                        if (e.message.includes('proposes Rule') || e.message.includes('📝')) {
                            eventClass += ' proposal';
                        } else if (e.message.includes('VOTE RESULT') || e.message.includes('📊')) {
                            eventClass += ' vote-result';
                        } else if (e.message.includes('ADOPTED') || e.message.includes('DEFEATED') || e.message.includes('✅') || e.message.includes('❌')) {
                            eventClass += ' important';
                        }
                        return `<div class="${eventClass}">[${e.timestamp}] ${e.message}</div>`;
                    }).join('');
                    document.getElementById('events').innerHTML = eventsHtml;
                    
                    // Check winner
                    if (data.winner) {
                        document.getElementById('winner').innerHTML = 
                            `<div class="winner">🏆 ${data.winner} WINS! 🏆</div>`;
                    }
                });
        }
    </script>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>🎮 Proper Nomic Game</h1>
            <p><strong>Multi-Instance Ollama • Validated Rules • Structured Proposals</strong></p>
            <button onclick="startGame()">Start New Game</button>
            <div id="status">Ready to start</div>
            <p><strong>Turn:</strong> <span id="turn">0</span> | <strong>Current Player:</strong> <span id="current">-</span></p>
        </div>
        
        <div id="winner"></div>
        
        <div id="proposal-section" style="display: none;" class="card">
            <h2>📝 Current Proposal</h2>
            <div id="current-proposal"></div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>👥 Players</h2>
                <div id="players">No game started</div>
            </div>
            
            <div class="card">
                <h2>💬 Game Events</h2>
                <div id="events" style="height: 400px; overflow-y: auto;">No events yet</div>
            </div>
        </div>
        
        <div class="triple">
            <div class="card">
                <h3>🔒 Immutable Rules</h3>
                <div id="immutable-rules" style="max-height: 350px; overflow-y: auto;">No rules loaded</div>
            </div>
            <div class="card">
                <h3>📝 Mutable Rules</h3>
                <div id="mutable-rules" style="max-height: 350px; overflow-y: auto;">No rules loaded</div>
            </div>
            <div class="card">
                <h3>⚙️ System Info</h3>
                <div style="font-size: 13px;">
                    <p><strong>Validator:</strong> llama3:8b on port 11435</p>
                    <p><strong>Player Instances:</strong> Ports 11436-11441</p>
                    <p><strong>Rule Numbering:</strong> System-controlled</p>
                    <p><strong>Response Parsing:</strong> Structured format</p>
                    <p><strong>Validation:</strong> Active rule checking</p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
    ''')

@app.route('/start', methods=['POST'])
def start_game():
    global game
    game = ProperNomicGame(6)
    game.start_game_thread()
    return jsonify({"status": "started"})

@app.route('/status')
def get_status():
    if not game:
        return jsonify({"error": "No game"})
    
    current_proposal_data = None
    if game.current_proposal:
        current_proposal_data = {
            "rule_text": game.current_proposal.rule_text,
            "explanation": game.current_proposal.explanation,
            "votes": game.current_proposal.votes,
            "next_number": game.next_rule_number
        }
    
    return jsonify({
        "turn": game.turn_number,
        "current_player": game.get_current_player().name if not game.game_over else None,
        "players": [
            {
                "id": p.id,
                "name": p.name,
                "role": p.role,
                "points": p.points,
                "model": p.model,
                "port": p.port,
                "current": p.id == game.get_current_player().id
            } for p in game.players
        ],
        "rules": {
            "immutable": [
                {
                    "id": r.id,
                    "text": r.text
                } for r in game.rules["immutable"]
            ],
            "mutable": [
                {
                    "id": r.id,
                    "text": r.text,
                    "effects": [{"description": e.description, "trigger": e.trigger, "action": e.action, "value": e.value} 
                               for e in r.effects] if hasattr(r, 'effects') else [],
                    "author": r.author if hasattr(r, 'author') else None,
                    "turn_added": r.turn_added if hasattr(r, 'turn_added') else None
                } for r in game.rules["mutable"]
            ]
        },
        "current_proposal": current_proposal_data,
        "events": game.events[-30:],
        "vote_history": game.vote_history[-10:],
        "turn_history": game.turn_history[-5:],
        "proposals_count": len(game.proposals),
        "voting_threshold": game.current_voting_threshold,
        "point_affecting_rules": [
            {
                "rule_id": r.id,
                "rule_text": r.text,
                "effects": [{"description": e.description, "trigger": e.trigger, "value": e.value} for e in r.effects if "point" in e.action]
            } for r in game.rules["mutable"] if hasattr(r, 'effects') and any("point" in e.action for e in r.effects)
        ],
        "winner": game.winner.name if game.winner else None
    })

@app.route('/stats')
def stats_page():
    """Beautiful HTML statistics page"""
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>Nomic Model Performance Statistics</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 1600px; 
            margin: 0 auto; 
        }
        .header {
            background: rgba(255,255,255,0.95);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .header h1 {
            color: #2c3e50;
            margin: 0 0 10px 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            color: #7f8c8d;
            font-size: 1.1em;
            margin: 0;
        }
        .stats-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
            margin-bottom: 20px;
        }
        .triple-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .card { 
            background: rgba(255,255,255,0.95); 
            padding: 25px; 
            border-radius: 15px; 
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .card h2 { 
            color: #2c3e50; 
            margin: 0 0 20px 0; 
            font-size: 1.4em;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        .card h3 { 
            color: #34495e; 
            margin: 0 0 15px 0; 
            font-size: 1.2em;
        }
        .model-card {
            background: rgba(255,255,255,0.9);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 5px solid #3498db;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .model-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }
        .model-card.elite { border-left-color: #f1c40f; background: linear-gradient(90deg, #fff9e6, #ffffff); }
        .model-card.competent { border-left-color: #2ecc71; background: linear-gradient(90deg, #e8f8f5, #ffffff); }
        .model-card.problematic { border-left-color: #e67e22; background: linear-gradient(90deg, #fdf2e9, #ffffff); }
        .model-card.elimination { border-left-color: #e74c3c; background: linear-gradient(90deg, #fdedec, #ffffff); }
        .current-player { border-left-color: #9b59b6; background: linear-gradient(90deg, #f4ecf7, #ffffff); }
        .model-name {
            font-weight: bold;
            font-size: 1.1em;
            color: #2c3e50;
            margin-bottom: 8px;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            padding: 3px 0;
            border-bottom: 1px solid #ecf0f1;
            font-size: 0.9em;
        }
        .metric-row:last-child { border-bottom: none; }
        .metric-label { color: #7f8c8d; }
        .metric-value { font-weight: 600; color: #2c3e50; }
        .tier-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .tier-elite { background: #f1c40f; color: #8b6508; }
        .tier-competent { background: #2ecc71; color: #ffffff; }
        .tier-problematic { background: #e67e22; color: #ffffff; }
        .tier-elimination { background: #e74c3c; color: #ffffff; }
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .summary-item {
            background: rgba(255,255,255,0.9);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #3498db;
        }
        .summary-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #3498db;
            margin: 0;
        }
        .summary-label {
            color: #7f8c8d;
            font-size: 0.9em;
            margin: 5px 0 0 0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .progress-bar {
            width: 100%;
            height: 6px;
            background: #ecf0f1;
            border-radius: 3px;
            overflow: hidden;
            margin: 5px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 3px;
            transition: width 0.3s ease;
        }
        .refresh-btn {
            background: linear-gradient(45deg, #3498db, #2ecc71);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: transform 0.2s;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            transform: scale(1.05);
        }
        .footer {
            text-align: center;
            color: rgba(255,255,255,0.8);
            margin-top: 30px;
            font-size: 0.9em;
        }
        @media (max-width: 1200px) {
            .stats-grid, .triple-grid { grid-template-columns: 1fr; }
        }
    </style>
    <script>
        function refreshStats() {
            location.reload();
        }
        
        function loadStats() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('main-content').innerHTML = 
                            '<div class="card"><h2>⚠️ No Game Running</h2><p>Start a game to view statistics.</p></div>';
                        return;
                    }
                    
                    updateSummaryStats(data.summary);
                    updateCurrentGame(data.current_game);
                    updateCrossGameRankings(data.cross_game_rankings);
                    updateModelTiers(data.model_tiers);
                });
        }
        
        function updateSummaryStats(summary) {
            document.getElementById('total-models').textContent = summary.total_models_tracked;
            document.getElementById('games-completed').textContent = summary.games_completed;
            document.getElementById('elite-count').textContent = summary.elite_count;
            document.getElementById('elimination-count').textContent = summary.elimination_candidates;
            document.getElementById('best-model').textContent = summary.best_model || 'None';
            document.getElementById('worst-model').textContent = summary.worst_model || 'None';
        }
        
        function updateCurrentGame(current_game) {
            const html = current_game.map(player => `
                <div class="model-card current-player">
                    <div class="model-name">${player.player_name} (${player.model})</div>
                    <div class="metric-row">
                        <span class="metric-label">Current Points:</span>
                        <span class="metric-value">${player.current_points}/100</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${player.current_points}%"></div>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Overall Score:</span>
                        <span class="metric-value">${player.overall_score.toFixed(1)}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Success Rate:</span>
                        <span class="metric-value">${player.success_rate.toFixed(1)}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Coherence:</span>
                        <span class="metric-value">${player.coherence.toFixed(1)}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Total Games:</span>
                        <span class="metric-value">${player.total_games}</span>
                    </div>
                </div>
            `).join('');
            document.getElementById('current-game').innerHTML = html;
        }
        
        function updateCrossGameRankings(rankings) {
            const html = rankings.slice(0, 10).map((model, index) => `
                <div class="model-card">
                    <div class="model-name">#${index + 1} ${model.model}</div>
                    <div class="metric-row">
                        <span class="metric-label">Overall Score:</span>
                        <span class="metric-value">${model.overall_score.toFixed(1)}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Games Played:</span>
                        <span class="metric-value">${model.games_played}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Success Rate:</span>
                        <span class="metric-value">${model.success_rate.toFixed(1)}%</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Coherence:</span>
                        <span class="metric-value">${model.coherence.toFixed(1)}</span>
                    </div>
                </div>
            `).join('');
            document.getElementById('cross-game-rankings').innerHTML = html;
        }
        
        function updateModelTiers(tiers) {
            updateTier('elite', tiers.elite, 'tier-elite');
            updateTier('competent', tiers.competent, 'tier-competent');
            updateTier('problematic', tiers.problematic, 'tier-problematic');
            updateTier('elimination', tiers.elimination_candidates, 'tier-elimination');
        }
        
        function updateTier(tierId, models, badgeClass) {
            if (models.length === 0) {
                document.getElementById(tierId).innerHTML = '<p style="color: #7f8c8d; font-style: italic;">No models in this tier</p>';
                return;
            }
            
            const html = models.map(model => `
                <div class="model-card ${tierId}">
                    <div class="model-name">
                        ${model.model}
                        <span class="tier-badge ${badgeClass}">${tierId}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Score:</span>
                        <span class="metric-value">${model.overall_score.toFixed(1)}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Games:</span>
                        <span class="metric-value">${model.games_played}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Errors:</span>
                        <span class="metric-value">${model.error_count}</span>
                    </div>
                </div>
            `).join('');
            document.getElementById(tierId).innerHTML = html;
        }
        
        // Auto-refresh every 10 seconds
        setInterval(loadStats, 10000);
        
        // Load stats when page loads
        window.onload = loadStats;
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 Nomic Model Performance Analytics</h1>
            <p>Comprehensive AI model evaluation and competitive analysis</p>
            <button class="refresh-btn" onclick="refreshStats()">🔄 Refresh Data</button>
        </div>
        
        <div id="main-content">
            <div class="summary-stats">
                <div class="summary-item">
                    <div class="summary-number" id="total-models">-</div>
                    <div class="summary-label">Models Tracked</div>
                </div>
                <div class="summary-item">
                    <div class="summary-number" id="games-completed">-</div>
                    <div class="summary-label">Games Completed</div>
                </div>
                <div class="summary-item">
                    <div class="summary-number" id="elite-count">-</div>
                    <div class="summary-label">Elite Models</div>
                </div>
                <div class="summary-item">
                    <div class="summary-number" id="elimination-count">-</div>
                    <div class="summary-label">At Risk</div>
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="card">
                    <h2>🎯 Current Game Players</h2>
                    <div id="current-game">Loading...</div>
                </div>
                
                <div class="card">
                    <h2>🏆 Cross-Game Rankings</h2>
                    <div id="cross-game-rankings">Loading...</div>
                </div>
            </div>
            
            <div class="card">
                <h2>📊 Performance Leaders</h2>
                <div class="stats-grid">
                    <div>
                        <strong>🥇 Best Model:</strong> <span id="best-model">-</span>
                    </div>
                    <div>
                        <strong>🥉 Worst Model:</strong> <span id="worst-model">-</span>
                    </div>
                </div>
            </div>
            
            <div class="triple-grid">
                <div class="card">
                    <h3>🌟 Elite Tier (75+ Score)</h3>
                    <div id="elite">Loading...</div>
                </div>
                
                <div class="card">
                    <h3>✅ Competent Tier (50-74 Score)</h3>
                    <div id="competent">Loading...</div>
                </div>
                
                <div class="card">
                    <h3>⚠️ Problematic Tier (25-49 Score)</h3>
                    <div id="problematic">Loading...</div>
                </div>
            </div>
            
            <div class="card">
                <h3>❌ Elimination Candidates (<25 Score)</h3>
                <div id="elimination">Loading...</div>
            </div>
        </div>
        
        <div class="footer">
            <p>🤖 Automated model performance tracking and competitive analysis for Nomic AI gameplay</p>
        </div>
    </div>
</body>
</html>
    ''')

@app.route('/api/stats')
def get_statistics_api():
    """API endpoint for statistics data"""
    if not game:
        return jsonify({"error": "No game"})
    
    # Get current game model assignments and performance
    current_game_stats = []
    for player in game.players:
        if player.assigned_model_metrics:
            metrics = player.assigned_model_metrics
            current_game_stats.append({
                "player_name": player.name,
                "model": player.model,
                "current_points": player.points,
                "overall_score": metrics.calculate_overall_score(),
                "success_rate": metrics.calculate_success_rate(),
                "coherence": metrics.calculate_average_coherence(),
                "engagement": metrics.calculate_strategic_engagement(),
                "tie_obsession": metrics.tie_related_proposals,
                "total_games": metrics.total_games,
                "games_won": metrics.games_won
            })
    
    # Get cross-game model statistics
    model_stats = game.performance_manager.get_model_statistics()
    
    # Sort models by overall score
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['overall_score'], reverse=True)
    
    # Create tier classifications
    model_tiers = {
        "elite": [],
        "competent": [],
        "problematic": [],
        "elimination_candidates": []
    }
    
    for model, stats in sorted_models:
        score = stats['overall_score']
        if score >= 75:
            model_tiers["elite"].append({"model": model, **stats})
        elif score >= 50:
            model_tiers["competent"].append({"model": model, **stats})
        elif score >= 25:
            model_tiers["problematic"].append({"model": model, **stats})
        else:
            model_tiers["elimination_candidates"].append({"model": model, **stats})
    
    return jsonify({
        "current_game": current_game_stats,
        "cross_game_rankings": [{"model": model, **stats} for model, stats in sorted_models],
        "model_tiers": model_tiers,
        "summary": {
            "total_models_tracked": len(model_stats),
            "games_completed": max([stats['games_played'] for stats in model_stats.values()] + [0]),
            "best_model": sorted_models[0][0] if sorted_models else None,
            "worst_model": sorted_models[-1][0] if sorted_models else None,
            "elite_count": len(model_tiers["elite"]),
            "elimination_candidates": len(model_tiers["elimination_candidates"])
        }
    })

if __name__ == "__main__":
    print("🎮 Starting Advanced Strategic Nomic Game with Deliberation & Competition")
    print("📱 Game Interface: http://127.0.0.1:8080")
    print("📊 Model Statistics: http://127.0.0.1:8080/stats")
    print("🧠 Enhanced Features:")
    print("   • 5-turn proposal deliberation loops for strategic thinking")
    print("   • 2-turn competitive voting deliberation with impact analysis")
    print("   • 8 distinct proposal categories with diversity enforcement")
    print("   • Multi-instance Ollama with performance-based model assignment")
    print("   • Comprehensive behavioral analysis and coherence tracking")
    print("   • Anti-groupthink competitive voting incentives")
    print("   • Cross-game model performance metrics and quality classification")
    print("   • Position-based strategic prompting and threat assessment")
    print("   • Semantic diversity blocking and category usage tracking")
    print("🎯 Expected Improvements:")
    print("   • End unanimous voting through strategic competition")
    print("   • Force true diversity in proposal types and mechanics")
    print("   • Generate innovative rules across different game aspects")
    print("   • Create actual strategic competition rather than cooperation")
    print("⏹️  Press Ctrl+C to stop")
    app.run(host='127.0.0.1', port=8080, debug=False)