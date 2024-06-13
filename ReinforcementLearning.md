# Reinforcement Learning

Reinforcement learning (RL) is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize some notion of cumulative reward. Unlike supervised learning, where the model is trained on a fixed dataset, reinforcement learning involves an agent interacting with the environment in real-time, learning from the consequences of its actions.

### Key Concepts in Reinforcement Learning

1. **Agent**:
   - The learner or decision-maker that interacts with the environment.

2. **Environment**:
   - Everything the agent interacts with. It receives the agent's actions and returns the next state and reward.

3. **State (s)**:
   - A representation of the current situation of the environment. The state includes all necessary information the agent needs to make a decision.

4. **Action (a)**:
   - Choices the agent can make at each state. The set of all possible actions in a state is called the action space.

5. **Reward (r)**:
   - A scalar feedback signal received after performing an action. The goal of the agent is to maximize the cumulative reward over time.

6. **Policy (π)**:
   - A strategy used by the agent to determine the next action based on the current state. Policies can be deterministic or stochastic.

7. **Value Function (V)**:
   - A function that estimates the expected cumulative reward starting from a state and following a particular policy thereafter. It helps the agent to evaluate how good a particular state is.

8. **Q-Value (Q)**:
   - Also known as the action-value function, it estimates the expected cumulative reward of taking a particular action in a given state and following a specific policy thereafter.

### Types of Reinforcement Learning

1. **Model-Free vs. Model-Based**:
   - **Model-Free**: The agent learns the policy or value function directly from experiences without modeling the environment (e.g., Q-learning, SARSA).
   - **Model-Based**: The agent builds a model of the environment and uses it to plan actions (e.g., Dynamic Programming).

2. **Value-Based vs. Policy-Based**:
   - **Value-Based**: The agent learns the value function or Q-value function to derive the policy (e.g., Q-learning).
   - **Policy-Based**: The agent directly learns the policy that maps states to actions (e.g., REINFORCE algorithm).
   - **Actor-Critic**: Combines both value-based and policy-based methods, where the actor updates the policy and the critic updates the value function.

### Common Algorithms in Reinforcement Learning

1. **Q-Learning**:
   - A model-free, value-based algorithm that seeks to learn the Q-value function. The policy is derived by selecting the action with the highest Q-value in each state.

2. **SARSA (State-Action-Reward-State-Action)**:
   - Similar to Q-learning but updates the Q-value based on the action actually taken by the policy, making it an on-policy algorithm.

3. **Deep Q-Networks (DQN)**:
   - Combines Q-learning with deep neural networks to handle large state and action spaces. Uses experience replay and target networks to stabilize training.

4. **REINFORCE**:
   - A policy-based, model-free algorithm that uses Monte Carlo methods to update the policy based on the total reward obtained in episodes.

5. **Actor-Critic**:
   - An architecture that includes both a policy network (actor) and a value network (critic). The actor updates the policy directly, and the critic estimates the value function to reduce variance.