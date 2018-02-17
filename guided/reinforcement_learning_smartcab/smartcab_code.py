
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Reinforcement Learning
# ## Project: Train a Smartcab to Drive
# 
# Welcome to the fourth project of the Machine Learning Engineer Nanodegree! In this notebook, template code has already been provided for you to aid in your analysis of the *Smartcab* and your implemented learning algorithm. You will not need to modify the included code beyond what is requested. There will be questions that you must answer which relate to the project and the visualizations provided in the notebook. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide in `agent.py`.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# -----
# 
# ## Getting Started
# In this project, you will work towards constructing an optimized Q-Learning driving agent that will navigate a *Smartcab* through its environment towards a goal. Since the *Smartcab* is expected to drive passengers from one location to another, the driving agent will be evaluated on two very important metrics: **Safety** and **Reliability**. A driving agent that gets the *Smartcab* to its destination while running red lights or narrowly avoiding accidents would be considered **unsafe**. Similarly, a driving agent that frequently fails to reach the destination in time would be considered **unreliable**. Maximizing the driving agent's **safety** and **reliability** would ensure that *Smartcabs* have a permanent place in the transportation industry.
# 
# **Safety** and **Reliability** are measured using a letter-grade system as follows:
# 
# | Grade 	| Safety 	| Reliability 	|
# |:-----:	|:------:	|:-----------:	|
# |   A+  	|  Agent commits no traffic violations,<br/>and always chooses the correct action. | Agent reaches the destination in time<br />for 100% of trips. |
# |   A   	|  Agent commits few minor traffic violations,<br/>such as failing to move on a green light. | Agent reaches the destination on time<br />for at least 90% of trips. |
# |   B   	| Agent commits frequent minor traffic violations,<br/>such as failing to move on a green light. | Agent reaches the destination on time<br />for at least 80% of trips. |
# |   C   	|  Agent commits at least one major traffic violation,<br/> such as driving through a red light. | Agent reaches the destination on time<br />for at least 70% of trips. |
# |   D   	| Agent causes at least one minor accident,<br/> such as turning left on green with oncoming traffic.       	| Agent reaches the destination on time<br />for at least 60% of trips. |
# |   F   	|  Agent causes at least one major accident,<br />such as driving through a red light with cross-traffic.      	| Agent fails to reach the destination on time<br />for at least 60% of trips. |
# 
# To assist evaluating these important metrics, you will need to load visualization code that will be used later on in the project. Run the code cell below to import this code which is required for your analysis.

# In[2]:

# Import the visualization code
import visuals as vs

# Pretty display for notebooks
get_ipython().magic('matplotlib inline')


# ### Understand the World
# Before starting to work on implementing your driving agent, it's necessary to first understand the world (environment) which the *Smartcab* and driving agent work in. One of the major components to building a self-learning agent is understanding the characteristics about the agent, which includes how the agent operates. To begin, simply run the `agent.py` agent code exactly how it is -- no need to make any additions whatsoever. Let the resulting simulation run for some time to see the various working components. Note that in the visual simulation (if enabled), the **white vehicle** is the *Smartcab*.

# ### Question 1
# In a few sentences, describe what you observe during the simulation when running the default `agent.py` agent code. Some things you could consider:
# - *Does the Smartcab move at all during the simulation?*
# - *What kind of rewards is the driving agent receiving?*
# - *How does the light changing color affect the rewards?*  
# 
# **Hint:** From the `/smartcab/` top-level directory (where this notebook is located), run the command 
# ```bash
# 'python smartcab/agent.py'
# ```

# **Answer:** Right now, the smartcab stays stationary during the simulation. The agent receives positive rewards when the intersection it is at has a red light (ie. it is not permitted to move ahead), and it receives negative rewards when there is a green light at the intersection (it should be moving ahead).

# ### Understand the Code
# In addition to understanding the world, it is also necessary to understand the code itself that governs how the world, simulation, and so on operate. Attempting to create a driving agent would be difficult without having at least explored the *"hidden"* devices that make everything work. In the `/smartcab/` top-level directory, there are two folders: `/logs/` (which will be used later) and `/smartcab/`. Open the `/smartcab/` folder and explore each Python file included, then answer the following question.

# ### Question 2
# - *In the *`agent.py`* Python file, choose three flags that can be set and explain how they change the simulation.*
# - *In the *`environment.py`* Python file, what Environment class function is called when an agent performs an action?*
# - *In the *`simulator.py`* Python file, what is the difference between the *`'render_text()'`* function and the *`'render()'`* function?*
# - *In the *`planner.py`* Python file, will the *`'next_waypoint()`* function consider the North-South or East-West direction first?*

# **Answer:** Epsilon is the random exploration factor, and describes how often the agent will deviate from it's learned behaviour, and execute a random move to explore the environment (and thus potentially learn new things about the environment). Alpha is the learning factor, and determines how quickly the agent will learn from the rewards and punishments it sees in the simulation. Therefore a higher alpha will cause given rewards and punishments to change the behaviour of the agent quicker. Lastly, the 'learning' variable simply determines whether the agent will learn at all from the environment, or if it will just act randomly.
# 
# The 'act' method is the Environment class function called when an agent performs an action.
# 
# The render() function is what displays what is seen in the graphical user interface (GUI), and the render_text() function is what displays what is seen in the command prompt.
# 
# It can be seen in the next_waypoint() function the East-West directions are considered first in the execution of the function. The directions are only executed if it is known that the car's location isn't in line, column-wise, with the destination.
# 

# -----
# ## Implement a Basic Driving Agent
# 
# The first step to creating an optimized Q-Learning driving agent is getting the agent to actually take valid actions. In this case, a valid action is one of `None`, (do nothing) `'left'` (turn left), `right'` (turn right), or `'forward'` (go forward). For your first implementation, navigate to the `'choose_action()'` agent function and make the driving agent randomly choose one of these actions. Note that you have access to several class variables that will help you write this functionality, such as `'self.learning'` and `'self.valid_actions'`. Once implemented, run the agent file and simulation briefly to confirm that your driving agent is taking a random action each time step.

# ### Basic Agent Simulation Results
# To obtain results from the initial simulation, you will need to adjust following flags:
# - `'enforce_deadline'` - Set this to `True` to force the driving agent to capture whether it reaches the destination in time.
# - `'update_delay'` - Set this to a small value (such as `0.01`) to reduce the time between steps in each trial.
# - `'log_metrics'` - Set this to `True` to log the simluation results as a `.csv` file in `/logs/`.
# - `'n_test'` - Set this to `'10'` to perform 10 testing trials.
# 
# Optionally, you may disable to the visual simulation (which can make the trials go faster) by setting the `'display'` flag to `False`. Flags that have been set here should be returned to their default setting when debugging. It is important that you understand what each flag does and how it affects the simulation!
# 
# Once you have successfully completed the initial simulation (there should have been 20 training trials and 10 testing trials), run the code cell below to visualize the results. Note that log files are overwritten when identical simulations are run, so be careful with what log file is being loaded!
# Run the agent.py file after setting the flags from projects/smartcab folder instead of projects/smartcab/smartcab.
# 

# In[3]:

# Load the 'sim_no-learning' log file from the initial simulation results
vs.plot_trials('sim_no-learning.csv')


# ### Question 3
# Using the visualization above that was produced from your initial simulation, provide an analysis and make several observations about the driving agent. Be sure that you are making at least one observation about each panel present in the visualization. Some things you could consider:
# - *How frequently is the driving agent making bad decisions? How many of those bad decisions cause accidents?*
# - *Given that the agent is driving randomly, does the rate of reliability make sense?*
# - *What kind of rewards is the agent receiving for its actions? Do the rewards suggest it has been penalized heavily?*
# - *As the number of trials increases, does the outcome of results change significantly?*
# - *Would this Smartcab be considered safe and/or reliable for its passengers? Why or why not?*

# **Answer:**
# From the above simulation it can be seen that the agent made bad decisions quite often, from the first graph it seems that on average these bad choices are made aroudn 40% of the time. The graph also shows that accidents occur at a frequency close to 10%. If we wanted to know to what degree the bad decisions are causing accidents, we could find the probability of an accident occuring, given that a bad choice was made. From the frequencies detailed earlier, this would amount to around 0.1/0.4 = 0.25. Therefore it could be estimated that 25% of the bad actions made, caused an accident of some sort.
# 
# The rate of reliability relates to how well the agent reached it's destination. Given that the agent above acted randomly, it has no concept of destination, and will not change it's strategy regardless of where it is on the map. Therefore the rate of reliability does not make sense for this agent.
# 
# Looking at the average rolling reward per action (ARRPA) graph, it can be seen that it is substantially negative rewards that the agent is generally receiving. The ARRPA can be seen to range from -4 to -6 throughout the trials. The relatively large magnitude negative rewards do indeed suggest the agent has been penalized heavily.
# 
# Surprisingly, there actually seems to be some noticeable trends in the data with increasing trials. This is unexpected given the random nature of the agent, however it could be coincidental, given the relatively low number of trials conducted. From the first graph a relatively steady drop of 10% can be seen from 1st to the 10th trial. The frequency of major violations also shows a similar trend. Looking at the ARRPA graph, we see a noticeable increase in reward from the 14th to 18th trials. However, these trends seem to change erratically with different runs, suggesting that they are indeed just coincidental.
# 
# 

# -----
# ## Inform the Driving Agent
# The second step to creating an optimized Q-learning driving agent is defining a set of states that the agent can occupy in the environment. Depending on the input, sensory data, and additional variables available to the driving agent, a set of states can be defined for the agent so that it can eventually *learn* what action it should take when occupying a state. The condition of `'if state then action'` for each state is called a **policy**, and is ultimately what the driving agent is expected to learn. Without defining states, the driving agent would never understand which action is most optimal -- or even what environmental variables and conditions it cares about!

# ### Identify States
# Inspecting the `'build_state()'` agent function shows that the driving agent is given the following data from the environment:
# - `'waypoint'`, which is the direction the *Smartcab* should drive leading to the destination, relative to the *Smartcab*'s heading.
# - `'inputs'`, which is the sensor data from the *Smartcab*. It includes 
#   - `'light'`, the color of the light.
#   - `'left'`, the intended direction of travel for a vehicle to the *Smartcab*'s left. Returns `None` if no vehicle is present.
#   - `'right'`, the intended direction of travel for a vehicle to the *Smartcab*'s right. Returns `None` if no vehicle is present.
#   - `'oncoming'`, the intended direction of travel for a vehicle across the intersection from the *Smartcab*. Returns `None` if no vehicle is present.
# - `'deadline'`, which is the number of actions remaining for the *Smartcab* to reach the destination before running out of time.

# ### Question 4
# *Which features available to the agent are most relevant for learning both **safety** and **efficiency**? Why are these features appropriate for modeling the *Smartcab* in the environment? If you did not choose some features, why are those features* not *appropriate? Please note that whatever features you eventually choose for your agent's state, must be argued for here. That is: your code in agent.py should reflect the features chosen in this answer.
# *
# 
# NOTE: You are not allowed to engineer new features for the smartcab. 

# **Answer:** The 'waypoint' variable is relevant for efficiency, as it details the method in which the agent can reach it's destination. It makes sense that the agent will increase it's efficiency (reliability) if it is able to follow the waypoints as best as possible. 
# 
# Looking at the 'inputs' variables:
# 
# The colour of the light at the intersection is highly relevant for safety, as the agent will need to make sure it stops when required at an intersection, given the colour of the traffic light. 
# 
# The 'left' feature is relevant for efficiency and safety, since the smartcab needs to know whether there are cars approaching from the left, if it wants to turn right at a red light. Simply stopping at a red light and never considering a right turn would cause a loss in efficiency. 
# 
# The 'right' feature is not relevant for either of safety or efficiency, as according to the traffic rules detailed for the city, it is never needed to know the direction of travel of a vehicle to the right of the agent. 
# 
# The 'oncoming' feature is important and relevant for safety, since the agent needs to know the direction of oncoming traffic if it wants to make a left turn on a green light.
# 
# The 'deadline' feature does not seem to be relevant for either of safety or efficiency - it would be expected that the agent would make it's decisions entirely on the situation it is in, according to the features above. It would not be expected that the agent change it's behaviour in any way, given an abundance or lack of actions remaining before the deadline.
# 
# 

# ### Define a State Space
# When defining a set of states that the agent can occupy, it is necessary to consider the *size* of the state space. That is to say, if you expect the driving agent to learn a **policy** for each state, you would need to have an optimal action for *every* state the agent can occupy. If the number of all possible states is very large, it might be the case that the driving agent never learns what to do in some states, which can lead to uninformed decisions. For example, consider a case where the following features are used to define the state of the *Smartcab*:
# 
# `('is_raining', 'is_foggy', 'is_red_light', 'turn_left', 'no_traffic', 'previous_turn_left', 'time_of_day')`.
# 
# How frequently would the agent occupy a state like `(False, True, True, True, False, False, '3AM')`? Without a near-infinite amount of time for training, it's doubtful the agent would ever learn the proper action!

# ### Question 5
# *If a state is defined using the features you've selected from **Question 4**, what would be the size of the state space? Given what you know about the environment and how it is simulated, do you think the driving agent could learn a policy for each possible state within a reasonable number of training trials?*  
# **Hint:** Consider the *combinations* of features to calculate the total number of states!

# **Answer:**
# The 'waypoint' feature has 3 possible values (right, forward, left), 
# The 'light' feature has 2 possible values (red, green), 
# The 'left' feature has 4 possible values (right, forward, left, none), 
# The 'oncoming' features has 4 possible values (right, forward, left, none). 
# 
# Therefore, the total size of the state space is 3 x 4 x 2 x 4  = 96. It seems likely that the driver can learn a policy for each of the 96 states, given a reasonable amount of training trials.

# ### Update the Driving Agent State
# For your second implementation, navigate to the `'build_state()'` agent function. With the justification you've provided in **Question 4**, you will now set the `'state'` variable to a tuple of all the features necessary for Q-Learning. Confirm your driving agent is updating its state by running the agent file and simulation briefly and note whether the state is displaying. If the visual simulation is used, confirm that the updated state corresponds with what is seen in the simulation.
# 
# **Note:** Remember to reset simulation flags to their default setting when making this observation!

# -----
# ## Implement a Q-Learning Driving Agent
# The third step to creating an optimized Q-Learning agent is to begin implementing the functionality of Q-Learning itself. The concept of Q-Learning is fairly straightforward: For every state the agent visits, create an entry in the Q-table for all state-action pairs available. Then, when the agent encounters a state and performs an action, update the Q-value associated with that state-action pair based on the reward received and the iterative update rule implemented. Of course, additional benefits come from Q-Learning, such that we can have the agent choose the *best* action for each state based on the Q-values of each state-action pair possible. For this project, you will be implementing a *decaying,* $\epsilon$*-greedy* Q-learning algorithm with *no* discount factor. Follow the implementation instructions under each **TODO** in the agent functions.
# 
# Note that the agent attribute `self.Q` is a dictionary: This is how the Q-table will be formed. Each state will be a key of the `self.Q` dictionary, and each value will then be another dictionary that holds the *action* and *Q-value*. Here is an example:
# 
# ```
# { 'state-1': { 
#     'action-1' : Qvalue-1,
#     'action-2' : Qvalue-2,
#      ...
#    },
#   'state-2': {
#     'action-1' : Qvalue-1,
#      ...
#    },
#    ...
# }
# ```
# 
# Furthermore, note that you are expected to use a *decaying* $\epsilon$ *(exploration) factor*. Hence, as the number of trials increases, $\epsilon$ should decrease towards 0. This is because the agent is expected to learn from its behavior and begin acting on its learned behavior. Additionally, The agent will be tested on what it has learned after $\epsilon$ has passed a certain threshold (the default threshold is 0.05). For the initial Q-Learning implementation, you will be implementing a linear decaying function for $\epsilon$.

# ### Q-Learning Simulation Results
# To obtain results from the initial Q-Learning implementation, you will need to adjust the following flags and setup:
# - `'enforce_deadline'` - Set this to `True` to force the driving agent to capture whether it reaches the destination in time.
# - `'update_delay'` - Set this to a small value (such as `0.01`) to reduce the time between steps in each trial.
# - `'log_metrics'` - Set this to `True` to log the simluation results as a `.csv` file and the Q-table as a `.txt` file in `/logs/`.
# - `'n_test'` - Set this to `'10'` to perform 10 testing trials.
# - `'learning'` - Set this to `'True'` to tell the driving agent to use your Q-Learning implementation.
# 
# In addition, use the following decay function for $\epsilon$:
# 
# $$ \epsilon_{t+1} = \epsilon_{t} - 0.05, \hspace{10px}\textrm{for trial number } t$$
# 
# If you have difficulty getting your implementation to work, try setting the `'verbose'` flag to `True` to help debug. Flags that have been set here should be returned to their default setting when debugging. It is important that you understand what each flag does and how it affects the simulation! 
# 
# Once you have successfully completed the initial Q-Learning simulation, run the code cell below to visualize the results. Note that log files are overwritten when identical simulations are run, so be careful with what log file is being loaded!

# In[4]:

# Load the 'sim_default-learning' file from the default Q-Learning simulation
vs.plot_trials('sim_default-learning.csv')


# ### Question 6
# Using the visualization above that was produced from your default Q-Learning simulation, provide an analysis and make observations about the driving agent like in **Question 3**. Note that the simulation should have also produced the Q-table in a text file which can help you make observations about the agent's learning. Some additional things you could consider:  
# - *Are there any observations that are similar between the basic driving agent and the default Q-Learning agent?*
# - *Approximately how many training trials did the driving agent require before testing? Does that number make sense given the epsilon-tolerance?*
# - *Is the decaying function you implemented for $\epsilon$ (the exploration factor) accurately represented in the parameters panel?*
# - *As the number of training trials increased, did the number of bad actions decrease? Did the average reward increase?*
# - *How does the safety and reliability rating compare to the initial driving agent?*

# **Answer:** 
# A distinctly noticeable difference between the previously implemented basic driving agent, and the default Q-learning agent, is that it can be seen that the Q-learning agent shows improvements in performance that are much more pronounced. We can see this with the large drop in the amount of bad actions made, of about 20%. A larger decrease in seen in the amount of accidents that occur as well. In addition, we see a more pronounced and sizeable rolling average reward per action increase, of close to 4 units and a notable increase in the rolling rate of reliability. 
# 
# The safety and reliability ratings both are an improvement over the previous implementation, with a D rating for safety and an A+ rating for reliability. Despite the improvements, the safety rating shows that the smartcab's policy for each state needs to be improved, in order to get an acceptable agent. Iterating over different values for the learning rate and the epsilon tolerance could be investigated. In addition, the way in which epsilon decays could be adjusted to provide increases in performance. Given the upward trend in the rolling reward in the latter stages of the trial seen in the graph above, it also seems likely that a larger amount of training trials could lead to improved performance. This would be linked to the epsilon tolerance and epsilon decay.

# -----
# ## Improve the Q-Learning Driving Agent
# The third step to creating an optimized Q-Learning agent is to perform the optimization! Now that the Q-Learning algorithm is implemented and the driving agent is successfully learning, it's necessary to tune settings and adjust learning paramaters so the driving agent learns both **safety** and **efficiency**. Typically this step will require a lot of trial and error, as some settings will invariably make the learning worse. One thing to keep in mind is the act of learning itself and the time that this takes: In theory, we could allow the agent to learn for an incredibly long amount of time; however, another goal of Q-Learning is to *transition from experimenting with unlearned behavior to acting on learned behavior*. For example, always allowing the agent to perform a random action during training (if $\epsilon = 1$ and never decays) will certainly make it *learn*, but never let it *act*. When improving on your Q-Learning implementation, consider the implications it creates and whether it is logistically sensible to make a particular adjustment.

# ### Improved Q-Learning Simulation Results
# To obtain results from the initial Q-Learning implementation, you will need to adjust the following flags and setup:
# - `'enforce_deadline'` - Set this to `True` to force the driving agent to capture whether it reaches the destination in time.
# - `'update_delay'` - Set this to a small value (such as `0.01`) to reduce the time between steps in each trial.
# - `'log_metrics'` - Set this to `True` to log the simluation results as a `.csv` file and the Q-table as a `.txt` file in `/logs/`.
# - `'learning'` - Set this to `'True'` to tell the driving agent to use your Q-Learning implementation.
# - `'optimized'` - Set this to `'True'` to tell the driving agent you are performing an optimized version of the Q-Learning implementation.
# 
# Additional flags that can be adjusted as part of optimizing the Q-Learning agent:
# - `'n_test'` - Set this to some positive number (previously 10) to perform that many testing trials.
# - `'alpha'` - Set this to a real number between 0 - 1 to adjust the learning rate of the Q-Learning algorithm.
# - `'epsilon'` - Set this to a real number between 0 - 1 to adjust the starting exploration factor of the Q-Learning algorithm.
# - `'tolerance'` - set this to some small value larger than 0 (default was 0.05) to set the epsilon threshold for testing.
# 
# Furthermore, use a decaying function of your choice for $\epsilon$ (the exploration factor). Note that whichever function you use, it **must decay to **`'tolerance'`** at a reasonable rate**. The Q-Learning agent will not begin testing until this occurs. Some example decaying functions (for $t$, the number of trials):
# 
# $$ \epsilon = a^t, \textrm{for } 0 < a < 1 \hspace{50px}\epsilon = \frac{1}{t^2}\hspace{50px}\epsilon = e^{-at}, \textrm{for } 0 < a < 1 \hspace{50px} \epsilon = \cos(at), \textrm{for } 0 < a < 1$$
# You may also use a decaying function for $\alpha$ (the learning rate) if you so choose, however this is typically less common. If you do so, be sure that it adheres to the inequality $0 \leq \alpha \leq 1$.
# 
# If you have difficulty getting your implementation to work, try setting the `'verbose'` flag to `True` to help debug. Flags that have been set here should be returned to their default setting when debugging. It is important that you understand what each flag does and how it affects the simulation! 
# 
# Once you have successfully completed the improved Q-Learning simulation, run the code cell below to visualize the results. Note that log files are overwritten when identical simulations are run, so be careful with what log file is being loaded!

# In[5]:

# Load the 'sim_improved-learning' file from the improved Q-Learning simulation
vs.plot_trials('sim_improved-learning.csv')


# ### Question 7
# Using the visualization above that was produced from your improved Q-Learning simulation, provide a final analysis and make observations about the improved driving agent like in **Question 6**. Questions you should answer:  
# - *What decaying function was used for epsilon (the exploration factor)?*
# - *Approximately how many training trials were needed for your agent before begining testing?*
# - *What epsilon-tolerance and alpha (learning rate) did you use? Why did you use them?*
# - *How much improvement was made with this Q-Learner when compared to the default Q-Learner from the previous section?*
# - *Would you say that the Q-Learner results show that your driving agent successfully learned an appropriate policy?*
# - *Are you satisfied with the safety and reliability ratings of the *Smartcab*?*

# **Answer:** 
# I decided that a large number of trials would be required to get consistent good behaviour. For this, after iterating with many different combinations, I utilized an epsilon decay function 0.99^t and a learning rate of 0.1. This was able to provide A+ ratings for both categories on a consistent basis. Choosing this combination allowed for a large amount of trials (>500), and the low learning rate made sure that the agent needed to encounter certain rewards in certain states many, many times before it actually changed it's behaviour. This was done to allow higher performance in spite of the stochasticity involved with the simulation.
# 
# A significant improvement was made over the default Q-learner. Most notably the safety rating increased to A+, and the reliability grade was able to be maintained at A+ also. It must be noted however, that due to the changes to the epsilon decay function and tolerance, significantly higher amounts of training trials were ran. This required a larger computational overhead, and so this approach may not be ideal for more complex simulations.
# 
# Looking at the average rolling reward and seeing how it is well above zero in the latter trials, I would be confident that the Q-learning agent has learned an appropriate policy for the given task.

# ### Define an Optimal Policy
# 
# Sometimes, the answer to the important question *"what am I trying to get my agent to learn?"* only has a theoretical answer and cannot be concretely described. Here, however, you can concretely define what it is the agent is trying to learn, and that is the U.S. right-of-way traffic laws. Since these laws are known information, you can further define, for each state the *Smartcab* is occupying, the optimal action for the driving agent based on these laws. In that case, we call the set of optimal state-action pairs an **optimal policy**. Hence, unlike some theoretical answers, it is clear whether the agent is acting "incorrectly" not only by the reward (penalty) it receives, but also by pure observation. If the agent drives through a red light, we both see it receive a negative reward but also know that it is not the correct behavior. This can be used to your advantage for verifying whether the **policy** your driving agent has learned is the correct one, or if it is a **suboptimal policy**.

# ### Question 8
# 
# 1. Please summarize what the optimal policy is for the smartcab in the given environment. What would be the best set of instructions possible given what we know about the environment? 
#    _You can explain with words or a table, but you should thoroughly discuss the optimal policy._
# 
# 2. Next, investigate the `'sim_improved-learning.txt'` text file to see the results of your improved Q-Learning algorithm. _For each state that has been recorded from the simulation, is the **policy** (the action with the highest value) correct for the given state? Are there any states where the policy is different than what would be expected from an optimal policy?_ 
# 
# 3. Provide a few examples from your recorded Q-table which demonstrate that your smartcab learned the optimal policy. Explain why these entries demonstrate the optimal policy.
# 
# 4. Try to find at least one entry where the smartcab did _not_ learn the optimal policy.  Discuss why your cab may have not learned the correct policy for the given state.
# 
# Be sure to document your `state` dictionary below, it should be easy for the reader to understand what each state represents.

# **Answer:**
# 
# Optimal policy in pseudocode:
# 
# if light is red
# 
#     if waypoint is right and left is none
#     
#         turn right
#         
#     else
#     
#         stop
#         
# else if light is green
# 
#     if waypoint is left and oncoming is none
#     
#         turn left
#         
#     if waypoint is right
#     
#         turn right
#         
#     if waypoint is forward
#     
#         go forward
#     
#     else
#         
#         stop
#         
# 
# 
# In the following examples, the representation of states is as follows: (waypoint direction, light colour, left, oncoming)
# 
# 
# 1:
# ('right', 'green', None, 'left')
# 
#  -- forward : 0.33
#  
#  -- right : 1.66
#  
#  -- None : -2.18
#  
#  -- left : 0.33
# 
# Here, the agent has the highest Q value for turning right, which is the appropriate action given the waypoint direction and the green light.
# 
# 2: 
# ('forward', 'red', None, None)
# 
#  -- forward : -10.24
#  
#  -- right : 1.10
#  
#  -- None : 1.88
#  
#  -- left : -10.27
#  
# Here the agent correctly stops at a red light, given that the waypoint specifies moving forward.
# 
# 3:
# ('left', 'red', 'right', None)
# 
#  -- forward : -4.88
#  
#  -- right : 0.62
#  
#  -- None : 0.11
#  
#  -- left : -6.97
#  
# Here is an example where the agent doesn't follow the optimal policy. Despite the waypoint being to turn left, the agent has the highest Q value for turning right. This is a permitted move, given that the traffic from the left is also turning right, however is not the optimal action. Since right is in the opposite direction to left, the waypoint, it would be expected that doing so would take the agent further away from it's goal. 'None' would be the optimal action for this state. Given that the Q values for the 'right' and 'None' actions are quite close to zero, it seems that the agent most likely didn't explore this state enough, to learn the optimal action. A higher number of training trials would be expected to remedy this inoptimal action.

# -----
# ### Optional: Future Rewards - Discount Factor, `'gamma'`
# Curiously, as part of the Q-Learning algorithm, you were asked to **not** use the discount factor, `'gamma'` in the implementation. Including future rewards in the algorithm is used to aid in propagating positive rewards backwards from a future state to the current state. Essentially, if the driving agent is given the option to make several actions to arrive at different states, including future rewards will bias the agent towards states that could provide even more rewards. An example of this would be the driving agent moving towards a goal: With all actions and rewards equal, moving towards the goal would theoretically yield better rewards if there is an additional reward for reaching the goal. However, even though in this project, the driving agent is trying to reach a destination in the allotted time, including future rewards will not benefit the agent. In fact, if the agent were given many trials to learn, it could negatively affect Q-values!

# ### Optional Question 9
# *There are two characteristics about the project that invalidate the use of future rewards in the Q-Learning algorithm. One characteristic has to do with the *Smartcab* itself, and the other has to do with the environment. Can you figure out what they are and why future rewards won't work for this project?*

# **Answer:**
# 
# The environment is constantly changing. Therefore, it makes no sense to look ahead and see what will occur in a state that isn't the current one, because by the time the smartcab reaches that state, the environment may have completely changed, in such a way that makes the agent no longer end up in the state that was assigned a future reward.

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# In[ ]:



