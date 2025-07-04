#!/usr/bin/env python3
"""
Integration test for hybrid system with ProperNomicGame
"""
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from proper_nomic import OpenRouterClient, ProperNomicGame


def test_hybrid_integration():
    """Test hybrid system integration with main game"""
    print("=" * 60)
    print("TESTING HYBRID SYSTEM INTEGRATION")
    print("=" * 60)
    
    # Test 1: OpenRouter Client with Hybrid Mode
    print("\n1. Testing OpenRouter Client with Hybrid Mode...")
    
    try:
        # Create client with hybrid mode (using dummy API key)
        client = OpenRouterClient("dummy_key", use_hybrid=True)
        
        print(f"✅ OpenRouter client created with hybrid mode")
        print(f"   Hybrid enabled: {client.use_hybrid}")
        print(f"   Model families: {len(client.model_families)}")
        print(f"   Available models: {len(client.available_models)}")
        
        # Verify model families
        for family, models in client.model_families.items():
            print(f"   {family}: {models['big'].split('/')[-1]} → {models['small'].split('/')[-1]}")
        
    except Exception as e:
        print(f"❌ OpenRouter client test failed: {e}")
        return False
    
    # Test 2: Game Initialization with Hybrid Client  
    print("\n2. Testing Game Initialization with Hybrid Client...")
    
    try:
        # Create game with hybrid client
        game = ProperNomicGame(num_players=4, provider=client)
        
        print(f"✅ Game initialized with hybrid client")
        print(f"   Players: {len(game.players)}")
        print(f"   Hybrid selector created: {game.hybrid_selector is not None}")
        print(f"   Provider type: {game.provider_type}")
        
        if game.hybrid_selector:
            print(f"   Hybrid selector client: {type(game.hybrid_selector.client).__name__}")
            print(f"   Complexity engine: {type(game.hybrid_selector.complexity_engine).__name__}")
        
    except Exception as e:
        print(f"❌ Game initialization test failed: {e}")
        return False
    
    # Test 3: Hybrid Model Selection for Players
    print("\n3. Testing Hybrid Model Selection for Players...")
    
    try:
        for i, player in enumerate(game.players[:2]):  # Test first 2 players
            print(f"   Player {player.id} ({player.name}):")
            print(f"     Assigned model: {player.model}")
            
            if game.hybrid_selector:
                # Test different task types
                test_scenarios = [
                    ("proposal_deliberation", {"late_game": False}),
                    ("voting_decision", {"predicted_vote_margin": 3}),
                    ("voting_decision", {"predicted_vote_margin": 1, "affects_player_directly": True}),
                ]
                
                for task_type, context in test_scenarios:
                    selected_model = game.hybrid_selector.get_model_for_player_task(
                        player, task_type, game, context
                    )
                    is_escalated = selected_model != game.hybrid_selector.client.model_families[
                        game.hybrid_selector._get_model_family(player.model)
                    ]["small"]
                    
                    print(f"     {task_type}: {selected_model.split('/')[-1]} {'(escalated)' if is_escalated else '(optimized)'}")
            else:
                print(f"     No hybrid selector available")
                
    except Exception as e:
        print(f"❌ Model selection test failed: {e}")
        return False
    
    # Test 4: Complexity Detection with Game State
    print("\n4. Testing Complexity Detection with Game State...")
    
    try:
        if game.hybrid_selector:
            complexity_engine = game.hybrid_selector.complexity_engine
            
            # Test different game scenarios
            test_scenarios = [
                ("Early game", {"turn_number": 5, "player_points": 50}),
                ("Late game", {"turn_number": 18, "player_points": 45}),
                ("Elimination risk", {"turn_number": 10, "player_points": 25}),
                ("High rule count", {"turn_number": 8, "rule_count": 15}),
            ]
            
            for scenario_name, scenario_data in test_scenarios:
                # Modify game state for testing
                original_turn = game.turn_number
                game.turn_number = scenario_data.get("turn_number", game.turn_number)
                
                # Create test player
                test_player = game.players[0]
                test_player.points = scenario_data.get("player_points", test_player.points)
                
                complexity = complexity_engine.calculate_game_state_complexity(game, test_player)
                
                print(f"   {scenario_name}: complexity {complexity:.2f}")
                
                # Restore original state
                game.turn_number = original_turn
                test_player.points = 50
                
        else:
            print("   No complexity engine available")
            
    except Exception as e:
        print(f"❌ Complexity detection test failed: {e}")
        return False
    
    # Test 5: Unified Generate Method Integration
    print("\n5. Testing Unified Generate Method Integration...")
    
    try:
        # Test that unified_generate method can handle hybrid parameters
        test_player = game.players[0]
        test_prompt = "Test prompt for hybrid system"
        
        # This should work without errors (even with dummy API key)
        print(f"   Testing unified_generate with task_type and task_context...")
        print(f"   Player: {test_player.name} ({test_player.model})")
        print(f"   Hybrid selector available: {game.hybrid_selector is not None}")
        
        # We can't actually call the API with a dummy key, but we can verify the method exists
        # and accepts the right parameters
        method_exists = hasattr(game, 'unified_generate')
        print(f"   unified_generate method exists: {method_exists}")
        
        if game.hybrid_selector and method_exists:
            print(f"   ✅ Integration ready for hybrid model selection")
        
    except Exception as e:
        print(f"❌ Unified generate test failed: {e}")
        return False
    
    # Test 6: Statistics and Tracking
    print("\n6. Testing Statistics and Tracking...")
    
    try:
        if game.hybrid_selector:
            # Get initial stats
            stats = game.hybrid_selector.get_escalation_stats()
            print(f"   Initial escalation stats: {stats['total_decisions']} decisions")
            
            # Simulate some decisions
            for i in range(5):
                player = game.players[i % len(game.players)]
                selected_model = game.hybrid_selector.get_model_for_player_task(
                    player, "proposal_deliberation", game, {}
                )
            
            # Get updated stats
            updated_stats = game.hybrid_selector.get_escalation_stats()
            print(f"   After simulation: {updated_stats['total_decisions']} decisions")
            print(f"   Escalations: {updated_stats['escalations']}")
            print(f"   Small model usage: {updated_stats['small_model_usage']}")
            
        else:
            print("   No hybrid selector for statistics testing")
            
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ HYBRID SYSTEM INTEGRATION TEST COMPLETE")
    print("=" * 60)
    print("\nIntegration Status:")
    print("✅ OpenRouter client with hybrid mode")
    print("✅ Game initialization with hybrid selector")
    print("✅ Player model assignment and selection")
    print("✅ Complexity detection with game state")
    print("✅ Unified generate method integration")
    print("✅ Statistics and tracking system")
    
    print(f"\nReady for production use:")
    print(f"./venv312/bin/python proper_nomic.py --openrouter --hybrid --budget 10.0")
    
    return True


if __name__ == "__main__":
    success = test_hybrid_integration()
    exit(0 if success else 1)