class Civilization:
    def __init__(self, name, resources, military_strength, relations=None):
        self.name = name
        self.resources = resources
        self.military_strength = military_strength
        self.relations = relations or {}

    def establish_relation(self, other, relation_type):
        self.relations[other] = relation_type
        other.relations[self] = relation_type

    def declare_war(self, other):
        self.establish_relation(other, "War")

    def form_alliance(self, other):
        self.establish_relation(other, "Alliance")

    def negotiate_treaty(self, other, terms):
        if self.relations.get(other) == "Alliance":
            self.establish_relation(other, "Treaty")
            other.establish_relation(self, "Treaty")
            return True
        return False

    def trade(self, other, give, receive):
        if self.relations.get(other) in ["Alliance", "Treaty"]:
            if give in self.resources and receive in other.resources:
                self.resources.remove(give)
                other.resources.remove(receive)
                self.resources.append(receive)
                other.resources.append(give)
                return True
        return False


class DiplomacyEngine:
    def __init__(self, civilizations):
        self.civilizations = {civ.name: civ for civ in civilizations}

    def evaluate_action(self, actor, action, target):
        actor_civ = self.civilizations[actor]
        target_civ = self.civilizations[target]

        if action == "declare_war":
            return self.evaluate_war(actor_civ, target_civ)
        elif action == "form_alliance":
            return self.evaluate_alliance(actor_civ, target_civ)
        elif action == "negotiate_treaty":
            return self.evaluate_treaty(actor_civ, target_civ)
        elif action == "trade":
            return self.evaluate_trade(actor_civ, target_civ)

    def evaluate_war(self, actor, target):
        # Evaluate potential costs and benefits of war
        # Returns a score from -1.0 (very bad) to 1.0 (very good)
        
        # Military strength comparison
        strength_ratio = actor.military_strength / max(target.military_strength, 1)
        military_score = min(1.0, (strength_ratio - 1.0) / 2.0)  # Normalize to [-1, 1]
        
        # Current relationship penalty
        current_relation = actor.relations.get(target, "Neutral")
        relation_penalty = 0.0
        if current_relation == "Alliance":
            relation_penalty = -0.8  # Very bad to attack ally
        elif current_relation == "Treaty":
            relation_penalty = -0.4  # Bad to attack treaty partner
        
        # Resource considerations - target's resources attractiveness
        target_resource_value = len(target.resources) * 0.1
        resource_score = min(0.5, target_resource_value)
        
        # Overall evaluation
        total_score = (military_score * 0.6) + (resource_score * 0.2) + (relation_penalty * 0.2)
        return max(-1.0, min(1.0, total_score))

    def evaluate_alliance(self, actor, target):
        # Evaluate potential benefits of an alliance
        # Returns a score from -1.0 (very bad) to 1.0 (very good)
        
        # Current relationship consideration
        current_relation = actor.relations.get(target, "Neutral")
        if current_relation == "War":
            return -0.9  # Can't ally with enemy
        elif current_relation == "Alliance":
            return 0.0  # Already allied
        
        # Military strength compatibility - prefer roughly equal or stronger allies
        strength_ratio = target.military_strength / max(actor.military_strength, 1)
        strength_score = min(1.0, strength_ratio / 2.0)  # Stronger allies are better
        
        # Resource complementarity - different resources are beneficial
        actor_resources = set(actor.resources)
        target_resources = set(target.resources)
        unique_resources = len(target_resources - actor_resources)
        resource_score = min(0.5, unique_resources * 0.1)
        
        # Strategic value - combined military strength
        combined_strength = (actor.military_strength + target.military_strength) / 200.0
        strategic_score = min(0.3, combined_strength)
        
        total_score = (strength_score * 0.4) + (resource_score * 0.4) + (strategic_score * 0.2)
        return max(-1.0, min(1.0, total_score))

    def evaluate_treaty(self, actor, target):
        # Evaluate potential terms of a treaty
        # Returns a score from -1.0 (very bad) to 1.0 (very good)
        
        # Current relationship consideration
        current_relation = actor.relations.get(target, "Neutral")
        if current_relation == "War":
            return -0.7  # Difficult to negotiate treaty during war
        elif current_relation == "Treaty":
            return 0.1  # Already have treaty, slight benefit to maintain
        elif current_relation == "Alliance":
            return 0.8  # Good foundation for treaty
        
        # Military balance - treaties work better between balanced powers
        strength_diff = abs(actor.military_strength - target.military_strength)
        balance_score = max(0.0, 1.0 - (strength_diff / 100.0))
        
        # Mutual benefit from trade potential
        actor_resources = set(actor.resources)
        target_resources = set(target.resources)
        trade_potential = len(actor_resources.intersection(target_resources)) + \
                         len(actor_resources.symmetric_difference(target_resources))
        trade_score = min(0.4, trade_potential * 0.05)
        
        # Peace dividend - avoiding conflict costs
        peace_bonus = 0.3 if current_relation != "War" else 0.1
        
        total_score = (balance_score * 0.4) + (trade_score * 0.3) + (peace_bonus * 0.3)
        return max(-1.0, min(1.0, total_score))

    def evaluate_trade(self, actor, target):
        # Evaluate potential benefits of a trade
        # Returns a score from -1.0 (very bad) to 1.0 (very good)
        
        # Relationship requirement
        current_relation = actor.relations.get(target, "Neutral")
        if current_relation == "War":
            return -1.0  # Cannot trade with enemies
        
        # Relationship bonus
        relation_bonus = 0.0
        if current_relation == "Alliance":
            relation_bonus = 0.3
        elif current_relation == "Treaty":
            relation_bonus = 0.2
        
        # Resource availability and need assessment
        actor_resources = set(actor.resources)
        target_resources = set(target.resources)
        
        # Resources that target has but actor doesn't
        beneficial_imports = target_resources - actor_resources
        import_value = min(0.4, len(beneficial_imports) * 0.1)
        
        # Resources that actor has but target doesn't (export potential)
        export_potential = actor_resources - target_resources
        export_value = min(0.3, len(export_potential) * 0.08)
        
        # Overall trade viability
        if not beneficial_imports and not export_potential:
            trade_viability = -0.2  # No mutual benefit
        else:
            trade_viability = import_value + export_value
        
        total_score = trade_viability + relation_bonus
        return max(-1.0, min(1.0, total_score))

    def negotiate(self, actor, action, target, terms=None):
        actor_civ = self.civilizations[actor]
        target_civ = self.civilizations[target]

        if action == "declare_war":
            actor_civ.declare_war(target_civ)
        elif action == "form_alliance":
            actor_civ.form_alliance(target_civ)
        elif action == "negotiate_treaty":
            actor_civ.negotiate_treaty(target_civ, terms)
        elif action == "trade":
            actor_civ.trade(target_civ, *terms)


# Documentation and usage examples

# To create a new civilization:
civ1 = Civilization("Civilization 1", ["Gold", "Iron"], 100)
civ2 = Civilization("Civilization 2", ["Wood", "Stone"], 80)

# To establish diplomatic relations:
civ1.form_alliance(civ2)  # Civilizations 1 and 2 are now allies

# To declare war:
civ1.declare_war(civ2)  # Civilizations 1 and 2 are now at war

# To negotiate a treaty:
civ1.negotiate_treaty(civ2, ["No aggression", "Open borders"])

# To engage in trade:
civ1.trade(civ2, "Gold", "Wood")  # Civ1 gives Gold, Civ2 gives Wood

# To use the DiplomacyEngine:
engine = DiplomacyEngine([civ1, civ2])
engine.negotiate("Civilization 1", "form_alliance", "Civilization 2")