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
        pass

    def evaluate_alliance(self, actor, target):
        # Evaluate potential benefits of an alliance
        pass

    def evaluate_treaty(self, actor, target):
        # Evaluate potential terms of a treaty
        pass

    def evaluate_trade(self, actor, target):
        # Evaluate potential benefits of a trade
        pass

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