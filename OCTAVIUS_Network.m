close all

%% ====================================================================
%  ENVIRONMENT INITIALIZATION
%  ====================================================================

% Create OCTAVIUS fusion plasma environment
env = OCTAVIUS_Enviroment;
obs = getObservationInfo(env);
act = getActionInfo(env);

% Set random seed for reproducibility
lastrng = rng(07012001, "twister");

%% ====================================================================
%  TRAINING PARAMETERS
%  ====================================================================

lim = 1e4;                          % Maximum number of training episodes
Tlim = env.TT;                      % Total simulation time (s)
steplim = ceil(Tlim / env.Ts);      % Steps per episode

%% ====================================================================
%  NEURAL NETWORK ARCHITECTURE
%  ====================================================================

numhidden = 128;                    % Number of hidden units per layer

%% --------------------------------------------------------------------
%  Critic Network (Q-Value Function)
%  --------------------------------------------------------------------

% Observation pathway
obsPath = [
    featureInputLayer(prod(obs.Dimension), Name="obsInLyr")
    layerNormalizationLayer 
    fullyConnectedLayer(numhidden, Name="obsFC1")
    reluLayer(Name="obsReLU1")
    fullyConnectedLayer(numhidden, Name="obsFC2")
    reluLayer(Name="obsReLU2")
    fullyConnectedLayer(numhidden, Name="obsFC3")
    reluLayer(Name="obsReLU3")
];

% Action pathway
actPath = [
    featureInputLayer(prod(act.Dimension), Name="actInLyr")
    fullyConnectedLayer(numhidden, Name="actFC1")
    reluLayer(Name="actReLU1")
    fullyConnectedLayer(numhidden, Name="actFC2")
    reluLayer(Name="actReLU2")
    fullyConnectedLayer(numhidden, Name="actFC3")
    reluLayer(Name="actReLU3")
];

% Common pathway (combines observation and action)
commonPath = [
    concatenationLayer(1, 2, Name="concat")
    fullyConnectedLayer(numhidden, Name="commonFC1")
    reluLayer(Name="commonReLU1")
    fullyConnectedLayer(numhidden, Name="commonFC2")
    reluLayer(Name="commonReLU2")
    fullyConnectedLayer(numhidden, Name="commonFC3")
    reluLayer(Name="commonReLU3")
    fullyConnectedLayer(1, Name="QValue")
];

% Construct critic network
cNet = dlnetwork();
cNet = addLayers(cNet, obsPath);
cNet = addLayers(cNet, actPath);
cNet = addLayers(cNet, commonPath);

% Connect observation and action pathways to common pathway
cNet = connectLayers(cNet, "obsReLU3", "concat/in1");
cNet = connectLayers(cNet, "actReLU3", "concat/in2");

% Initialize network weights
cNet = initialize(cNet);
summary(cNet)

% Create critic function
critic = rlQValueFunction(cNet, obs, act, ...
    'ObservationInputNames', 'obsInLyr', 'ActionInputNames', 'actInLyr');

%% --------------------------------------------------------------------
%  Actor Network (Policy Function)
%  --------------------------------------------------------------------

aNet = [
    featureInputLayer(prod(obs.Dimension))
    layerNormalizationLayer 
    fullyConnectedLayer(numhidden)
    reluLayer(Name="actorReLU1")
    fullyConnectedLayer(numhidden)
    reluLayer(Name="actorReLU2")
    fullyConnectedLayer(numhidden)
    reluLayer(Name="actorReLU3")
    fullyConnectedLayer(prod(act.Dimension))
    sigmoidLayer                    % Sigmoid ensures actions in [0,1]
];

% Initialize actor network
aNet = dlnetwork(aNet);
aNet = initialize(aNet);
summary(aNet)

% Create actor function
actor = rlContinuousDeterministicActor(aNet, obs, act);

%% ====================================================================
%  DDPG AGENT CONFIGURATION
%  ====================================================================

% Network initialization options
init = rlAgentInitializationOptions('NumHiddenUnit', numhidden, ...
    'Normalization', 'rescale-zero-one');

% Critic optimizer settings
criticOpt = rlOptimizerOptions( ...
    'LearnRate', 5e-4, ...
    'GradientThreshold', 1, ...
    'L2RegularizationFactor', 1e-4, ...
    'Algorithm', 'adam' ...
);

% Actor optimizer settings
actorOpt = rlOptimizerOptions( ...
    'LearnRate', 1e-5, ...
    'GradientThreshold', 1, ...
    'L2RegularizationFactor', 1e-4, ...
    'Algorithm', 'adam' ...
);

% Experience buffer and training batch settings
size = 128;

% DDPG agent options
agent = rlDDPGAgentOptions( ...
    'SampleTime', env.Ts, ...
    'ExperienceBufferLength', 1e7, ...
    'ActorOptimizerOptions', actorOpt, ...
    'MiniBatchSize', size, ...
    'NumWarmStartSteps', size, ...
    'NumStepsToLookAhead', size, ...
    'NumEpoch', 1, ...
    'CriticOptimizerOptions', criticOpt, ...
    'TargetSmoothFactor', 1e-3, ...
    'DiscountFactor', 0.99 ...
);

% Set optimizer options
agent.ActorOptimizerOptions = actorOpt;
agent.CriticOptimizerOptions = criticOpt;

% Exploration noise settings
agent.NoiseOptions.StandardDeviation = 0.6;
agent.NoiseOptions.StandardDeviationDecayRate = 5e-6;

%% ====================================================================
%  TRAINING CONFIGURATION
%  ====================================================================

trainOpt = rlTrainingOptions( ...
    'MaxEpisodes', lim, ...
    'MaxStepsPerEpisode', steplim, ...
    'StopOnError', 'on', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'EvaluationStatistic', ...
    'StopTrainingValue', 3.85 * steplim, ...
    'ScoreAveragingWindowLength', 50, ...
    'SaveAgentCriteria', 'EpisodeFrequency', ...
    'SaveAgentValue', 50, ...
    'UseParallel', false ...
);

% Configure agent save options
agent.InfoToSave.ExperienceBuffer = true;
% agent.InfoToSave.PolicyState = true;
% agent.InfoToSave.Optimizer = true;
% agent.InfoToSave.Target = true;

% Set parallelization mode
trainOpts.ParallelizationOptions.Mode = "async";

% Evaluator for periodic performance assessment
evl = rlEvaluator("EvaluationFrequency", 50, "NumEpisodes", 10);

%% ====================================================================
%  VISUALIZATION SETUP
%  ====================================================================
% Comment out ti disable


% plotEe(env);                        % Electron energy plot
% plotEi(env);                        % Ion energy plot
% plotN(env);                         % Total particle density plot
% plotGamma(env);                     % Tritium fraction plot
% plotHeating(env);                 % Auxiliary heating plot (optional)
% plotFueling(env);                 % Fueling rates plot (optional)

%% ====================================================================
%  AGENT INITIALIZATION
%  ====================================================================

USE_PRE_TRAINED_MODEL = false;      % Set to true to continue training from saved agent

% Configure experience buffer reset behavior
agent.ResetExperienceBufferBeforeTraining = ~USE_PRE_TRAINED_MODEL;

if USE_PRE_TRAINED_MODEL
    % Load pre-trained agent and continue training
    sprintf('- Continue training pre-trained model');
    addpath('/Users/ianward/Desktop/Thesis/OCTAVIUS/savedAgents')
    load("AGENT.mat", "saved_agent")
    agent = saved_agent;
else
    % Create fresh DDPG agent
    agent = rlDDPGAgent(actor, critic, agent);
end

%% ====================================================================
%  TRAINING EXECUTION
%  ====================================================================

dotraining = 0;                     % Set to 0 to train, 1 to load saved agent

if dotraining == 0
    % Train the agent
    trainStat = train(agent, env, trainOpt, Evaluator=evl);
else
    % Load previously trained agent
    addpath('/Users/ianward/Desktop/Thesis/OCTAVIUS/savedAgents')
    load("AGENT.mat", "saved_agent")
    agent = saved_agent;
end

%% ====================================================================
%  SIMULATION AND EVALUATION
%  ====================================================================

% Configure simulation options
simOpt = rlSimulationOptions('MaxSteps', steplim);

% Run simulation with trained agent
exp = sim(env, agent, simOpt);