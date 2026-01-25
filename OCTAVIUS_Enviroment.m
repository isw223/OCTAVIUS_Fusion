classdef OCTAVIUS_Enviroment < rl.env.MATLABEnvironment
    
    properties(Constant)
        
        %% Fusion Reactivity Coefficients (H-S Bosch 1993)
        BG = 34.3827                    % Gamov constant
        C1 = 1.17392e-9                 % Reactivity coefficient 1
        C2 = 1.51361e-2                 % Reactivity coefficient 2
        C3 = 7.51886e-2                 % Reactivity coefficient 3
        C4 = 4.60643e-3                 % Reactivity coefficient 4
        C5 = 1.35e-2                    % Reactivity coefficient 5
        C6 = -1.06750e-4                % Reactivity coefficient 6
        C7 = 1.366e-5                   % Reactivity coefficient 7
        
        % ====================================================================
        %  SIMULATION CONTROL PARAMETERS
        %  ====================================================================
        %  This section controls the temporal behavior of the simulation and
        %  how target values transition during training/operation.
        %
        %  ADJUSTING THESE VALUES:
        %  - Ts: Smaller values = more accurate but slower simulation
        %  - TT: Longer simulations allow more training per episode
        %  - target_length: How long to track each target before switching
        %  - target_transition_time: Set to 1 for instant step changes,
        %    increase for smooth ramps between targets
        %  ====================================================================
        
        S_I_inj = 0                     % Wall impurity injection rate (1/m^3s)
                                      
        Ts = 0.1                        % Time step size (s)
                                        % Integration step for plasma dynamics
                                        % Typical range: 0.01 - 0.5 s
                                        % Smaller = more accurate, but slower training
        
        TT = 300                        % Total simulation time per episode (s)
                                        % Determines episode length
                                        % At Ts=0.1s, TT=300s gives 3000 steps/episode
        
        target_length = 800             % Target tracking duration (time steps)
                                        % Number of steps to maintain each target
                                        % At Ts=0.1s, 800 steps = 80 seconds
                                        % Increase for longer steady-state periods
        
        target_transition_time = 1      % Target transition duration (time steps)
                                        % Controls how targets change:
                                        %   = 1: Instant step changes (no ramp)
                                        %   > 1: Smooth transition over N steps
                                        % Example: 100 steps = 10s transition at Ts=0.1s
                                        % Set rampOn=false to disable transitions entirely
        %% Particle Masses (kg)
        m_e = 9.1096e-31                % Electron mass
        m_D = 3.3443254e-27             % Deuterium mass
        m_T = 5.008267217094e-27        % Tritium mass
        m_alpha = 6.6446573450e-27      % Alpha particle mass
        m_I = 1.4965078974e-26          % Beryllium ion mass
        m_s = 3.016029 * 1.660539040e-27 % Helium-3 mass
        mr = 1124656                    % Reduced mass of D and T

        %% Physical Constants and Conversions
        ev_to_k = 8.6173303e-5          % eV to Kelvin conversion factor
        e_charge = 1.6022e-19           % Elementary charge (C)
        epsilon_0 = 8.858e-12           % Vacuum permittivity (F/m)
        Q_alpha = 3.52e6                % Alpha particle energy (eV)

        %% Atomic Numbers
        ZD = 1                          % Deuterium proton number
        ZT = 1                          % Tritium proton number
        Z_s = 2                         % Helium-3 proton number
        Z_I = 4                         % Beryllium proton number
        
        %% Profile Factors
        gamma_t = 0                     % Radial temperature profile factor
        zeta_i = 1.1                    % Ion confinement factor
        zeta_e = 0.9                    % Electron confinement factor

        %% Heating and Energy Efficiency Parameters
        eta_ic = 0.9                    % Ion cyclotron heating efficiency
        eta_ec = 0.92                   % Electron cyclotron heating efficiency
        eta_nbi1 = 1                    % Neutral beam injection 1 efficiency
        eta_nbi2 = 0.95                 % Neutral beam injection 2 efficiency
        phi_nbi = 0.2                   % NBI power split to ions

        %% Fueling Efficiency Parameters
        gamma_DT = 0.9                  % D-T fuel mix fraction
        eta_D = 0.93                    % Deuterium fueling efficiency
        eta_DT = 1                      % D-T mix fueling efficiency

        %% ====================================================================
        %  PLASMA GEOMETRY AND MAGNETIC FIELD PARAMETERS
        %  ====================================================================
        %  These parameters define the physical characteristics of the tokamak
        %  device. Modify these values to simulate different machines or 
        %  experimental scenarios.
        %
        %  DEFAULT CONFIGURATION: ITER-like parameters (Scenario A from NF22)
        %
        %  TO ADAPT FOR YOUR TOKAMAK:
        %  1. Update geometric parameters (V, a, R, epsilon, kappa)
        %  2. Set magnetic field strengths (B_T, I_P)
        %  3. Adjust H_H based on expected confinement performance
        %  4. Verify epsilon = a/R (auto-calculated or manual)

        %% ====================================================================
        %  EXAMPLE TOKAMAK CONFIGURATIONS
        %  ====================================================================
        %
        % ITER (Default - Baseline Scenario- Only Tested Scenario):
        %   V = 837,  a = 2.0,  B_T = 5.3,   R = 6.2,   epsilon = 0.323
        %   kappa = 1.7,  I_P = 15,  H_H = 1.0
        %
        % JET:
        %   V = 100,  a = 1.25,  B_T = 3.45,  R = 2.96,  epsilon = 0.422
        %   kappa = 1.75,  I_P = 4.8,  H_H = 1.0
        %
        % DIII-D:
        %   V = 31.6,  a = 0.67,  B_T = 2.0,  R = 1.67,  epsilon = 0.401
        %   kappa = 1.8,  I_P = 2.0,  H_H = 1.0
        %
        % SPARC:
        %   V = 28.7,  a = 0.57,  B_T = 12.2,  R = 1.85,  epsilon = 0.308
        %   kappa = 1.97,  I_P = 8.7,  H_H = 0.7
        %
        % ARC:
        %   V = 157,  a = 1.1,  B_T = 9.2,  R = 3.3,  epsilon = 0.333
        %   kappa = 1.84,  I_P = 7.8,  H_H = 1.0
        %  ====================================================================


        V = 837                         % Plasma volume (m³)
                                        % Total volume enclosed by last closed flux surface
        
        a = 2                           % Minor radius (m)
                                        % Horizontal half-width of plasma cross-section
                                        % Measured at midplane from magnetic axis to separatrix
        
        B_T = 5.3                       % Toroidal magnetic field (T)
                                        % Field strength at major radius R
                                        % Higher B_T → better confinement, higher fusion power
        
        R = 6.2                         % Major radius (m)
                                        % Distance from tokamak center to plasma magnetic axis
                                        % Larger R → higher plasma volume, more neutron shielding
        
        epsilon = 0.3226                % Inverse aspect ratio (dimensionless)
                                        % epsilon = a/R
        
        kappa = 1.7                     % Elongation at 95% flux surface (dimensionless)
                                        % Ratio of plasma height to width (κ = b/a)
        
        I_P = 15                        % Plasma current (MA)
                                        % Total toroidal current flowing through plasma
                                        % Greenwald limit: n_G ∝ I_P/(πa²)
        
        H_H = 1                         % H-mode confinement enhancement factor (dimensionless)
                                        % Multiplier on IPB98(y,2) energy confinement scaling
                                        % H_H = 1.0 (standard H-mode)
                                        % H_H < 1.0 (L-mode or degraded confinement)
                                        % H_H > 1.0 (enhanced confinement)
                                        % Typical range: 0.7-1.5
                                        % Advanced scenarios may achieve H_H = 1.2-1.5
        
        %% Uncertainty Parameters
        phi_alpha = 0.15                % Alpha particle heating uncertainty

        %% Particle Confinement Time Multipliers
        k_alpha = 4                     % Alpha particle confinement multiplier
        k_D = 3                         % Deuterium confinement multiplier
        k_T = 2                         % Tritium confinement multiplier
        k_I = 6                         % Wall impurity confinement multiplier
        k_s = 4                         % Seeded impurity confinement multiplier
        
        %% Recycling and Sputtering Parameters
        % NOTE: Set f_I_sp to zero for simulations without impurities
        gamma_pfc = 0.5                 % Tritium fraction at plasma-facing components
        f_ref = 0.5                     % Particle reflection factor [0.2 - 0.9]
        f_eff = 0.1                     % Recycling efficiency factor [0.1 - 0.5]
        R_eff = 0.6                     % Recycling-to-source flux ratio [> 0.6]
        f_I_sp = 0.01                   % Sputtered wall impurity fraction
        
    end
    
    properties
        
        %% Target State Variables
        Ee_bar                          % Target electron energy density
        Ei_bar                          % Target ion energy density
        n_bar                           % Target total particle density
        gamma_bar                       % Target tritium fraction
        
        %% Current State Variables
        State                           % Current plasma state vector
        count                           % Time step counter
        target_count = 1                % Target update counter
        gamma                           % Current tritium fraction
        n_tot                           % Current total particle density
        
        %% Transition Control Variables
        use_logarithmic_transition      % Flag for logarithmic vs linear transition
        statei                          % Initial state
        transition_steps = 0            % Remaining transition steps
        step_gamma                      % Gamma transition step size
        step_Ee                         % Electron energy transition step size
        step_Ei                         % Ion energy transition step size
        step_n                          % Density transition step size
        stateSS                         % Steady-state reference

        %% Control Flags
        rampOn = true                   % Enable/disable target transitions
        
    end
    
    properties(Access = protected)
        
        %% Figure Handles for Real-Time Plotting
        Ei_figure                       % Ion energy plot
        Ee_figure                       % Electron energy plot
        n_tot_figure                    % Total density plot
        n_D_figure                      % Deuterium density plot
        n_T_figure                      % Tritium density plot
        n_I_figure                      % Wall impurity density plot
        n_alpha_figure                  % Alpha particle density plot
        gamma_figure                    % Tritium fraction plot
        fuelingFigure                   % Fueling rates plot
        heatingFigure                   % Heating powers plot
        
    end

    %% ====================================================================
    %  CORE ENVIRONMENT METHODS
    %  ====================================================================
    
    methods
        
        % -----------------------------------------------------------------
        % Constructor: Initialize the RL environment
        % -----------------------------------------------------------------
        function env = OCTAVIUS_Enviroment()
            
            % Define observation space (12 state variables)
            ObservationInfo = rlNumericSpec([12 1]);
            ObservationInfo.Name = 'Plasma States and Targets';
            ObservationInfo.Description = ['Observed Electron Energy Density, Target Electron Energy Density, ' ...
                'Observed Ion Energy Density, Target Ion Density, Total Particle Density, Tritium Particle Density, ' ...
                'Deuterium Particle Density, Wall Impurity Density, Alpha Particle Density, Electron Particle Density, ' ...
                'Target Total Particle Density, Observed Tritium Fraction, Target Tritium Fraction'];
    
            % Define action space (6 control actuators)
            ActionInfo = rlNumericSpec([6 1]);
            ActionInfo.Name = "Control Actuators";
            ActionInfo.Description = "Electron Heating, Ion Heating, D Fuel, T Fuel, D Injection, DT Injection";
            
            % Initialize parent environment
            env = env@rl.env.MATLABEnvironment(ObservationInfo, ActionInfo);
    
            % Set action limits (normalized between 0 and 1)
            env.ActionInfo.LowerLimit = [0; 0; 0; 0; 0; 0];
            env.ActionInfo.UpperLimit = [1; 1; 1; 1; 1; 1];
            
        end
        
        % -----------------------------------------------------------------
        % Feed-Forward Simulation: Compute steady-state target conditions
        % -----------------------------------------------------------------
        function [Ee, Ei, n, gamma] = feedForward(env, n_alpha, n_D, n_T, n_I, Ee, Ei, n_s)
            
            % Store current state as reference
            env.stateSS = [n_alpha; n_D; n_T; n_I; Ee; Ei; n_s];
            
            % Generate random fueling rates within realistic bounds
            fuelingD = unifrnd(0.3, 0.45);
            fuelingT = fuelingD + unifrnd(0.10, 0.15);
        
            % Compute control inputs with randomized initial conditions
            F = [unifrnd(0.25, 0.40); unifrnd(0.25, 0.40); fuelingD; fuelingT] .* ...
                [60e6; 60e6; 10e18; 10e18];
            
            Paux_e = F(1) / env.V;          % Electron auxiliary heating power density
            Paux_i = F(2) / env.V;          % Ion auxiliary heating power density
            S_D = F(3);                     % Deuterium source rate
            S_T = F(4);                     % Tritium source rate

            % Run forward simulation for 1250 time steps to reach steady state
            for i = 1:1250

                % Compute tritium fraction
                gammaFF = n_T / (n_D + n_T);
    
                % Compute average ion mass number
                M = 3 * gammaFF + 2 * (1 - gammaFF);
                
                % Calculate particle densities
                n_e_pre_step = n_D + n_T + 2*n_alpha + env.Z_I*n_I + env.Z_s*n_s;  % Electron density
                n_i = n_D + n_T + n_alpha + n_I + n_s;                             % Ion density
                n_tot_pre_step = n_e_pre_step + n_i;                               % Total density
    
                % Compute effective charge
                Z_eff = (n_D + n_T + 4*n_alpha + (env.Z_I^2)*n_I + (env.Z_s^2)*n_s) / n_e_pre_step;
    
                % Calculate temperatures from energy densities
                T_ion_J = ((2/3) * Ei / n_i) * (1 + env.gamma_t);                  % Ion temperature (J)
                T_ion_ev = (((2/3) * Ei / n_i) / env.e_charge) * (1 + env.gamma_t); % Ion temperature (eV)
                T_ion_kev = T_ion_ev / 1000;                                       % Ion temperature (keV)
                
                T_e_J = ((2/3) * Ee / n_e_pre_step) * (1 + env.gamma_t);           % Electron temperature (J)
                T_e_ev = (((2/3) * Ee / n_e_pre_step) / env.e_charge) * (1 + env.gamma_t); % Electron temperature (eV)
                T_e_kev = T_e_ev / 1000;                                           % Electron temperature (keV)
                T_e_k = T_e_ev / env.ev_to_k;                                      % Electron temperature (K)
                
                % Calculate Coulomb logarithm
                lamda_e = (1.24e7) * (T_e_k^1.5) / (Z_eff^2 * sqrt(n_e_pre_step)) / (1 + 3*env.gamma_t/2);
                lnAe = log(lamda_e);
    
                % Compute D-T fusion reactivity using Bosch-Hale parameterization
                omega = T_ion_kev / (1 - (T_ion_kev * (env.C2 + T_ion_kev * (env.C4 + env.C6*T_ion_kev))) / ...
                    (1 + T_ion_kev * (env.C3 + T_ion_kev * (env.C5 + env.C7*T_ion_kev))));
                
                zeta = (env.BG^2 / (4*omega))^(1/3);
                
                % D-T reactivity (m^3/s)
                sigma_v = env.C1 * omega * sqrt(zeta / (env.mr*T_ion_kev^3)) * exp(-3*zeta) * 1e-6;
                
                % D-T fusion reaction rate density (1/m^3s)
                S_alpha = gammaFF * (1 - gammaFF) * (n_D + n_T)^2 * sigma_v;
    
                % Calculate electron-ion collision time
                if n_s < 1e3
                    tau_ei = (((2*pi)^0.5 * 3*pi * env.epsilon_0^2 * T_e_J^1.5) / ...
                        (env.e_charge^4 * env.m_e^0.5 * lnAe)) * ...
                        (env.m_D/n_D + env.m_T/n_T + env.m_alpha/(4*n_alpha) + env.m_I/(n_I*env.Z_I^2)) / ...
                        (1 + 3*env.gamma_t/2);
    
                elseif n_I < 1e3
                    tau_ei = (((2*pi)^0.5 * 3*pi * env.epsilon_0^2 * T_e_J^1.5) / ...
                        (env.e_charge^4 * env.m_e^0.5 * lnAe)) * ...
                        (env.m_D/n_D + env.m_T/n_T + env.m_s/(n_s*env.Z_s^2) + env.m_alpha/(4*n_alpha)) / ...
                        (1 + 3*env.gamma_t/2);
                        
                elseif n_I < 1e3 && n_s < 1e3
                    tau_ei = (((2*pi)^0.5 * 3*pi * env.epsilon_0^2 * T_e_J^1.5) / ...
                        (env.e_charge^4 * env.m_e^0.5 * lnAe)) * ...
                        (env.m_D/n_D + env.m_T/n_T + env.m_alpha/(4*n_alpha)) / ...
                        (1 + 3*env.gamma_t/2);
                else
                    tau_ei = (((2*pi)^0.5 * 3*pi * env.epsilon_0^2 * T_e_J^1.5) / ...
                        (env.e_charge^4 * env.m_e^0.5 * lnAe)) * ...
                        (env.m_D/n_D + env.m_T/n_T + env.m_s/(n_s*env.Z_s^2) + ...
                        env.m_alpha/(4*n_alpha) + env.m_I/(n_I*env.Z_I^2)) / ...
                        (1 + 3*env.gamma_t/2);
                end
    
                % Calculate power sources and sinks
                P_alpha = (env.Q_alpha * env.e_charge) * S_alpha;  % Fusion power (W/m^3)
                P_ei = 1.5 * n_e_pre_step * ((T_e_J - T_ion_J) / tau_ei); % Collisional exchange (W/m^3)
                P_oh = 2.8e-9 * Z_eff * (env.I_P*1e6)^2 / (env.a^4 * T_e_kev^1.5); % Ohmic heating (W/m^3)
                P_br = (5.5e-37) * (n_D + n_T + 4*n_alpha + n_I*env.Z_I^2 + n_s*env.Z_s^2) * ...
                    n_e_pre_step * sqrt(T_e_kev);  % Bremsstrahlung (W/m^3)
                
                % Total heating power (bounded to avoid numerical issues)
                P = max(Paux_i + Paux_e - P_br + P_alpha + P_oh, 1e-3);
                
                % Energy confinement time (IPB98(y,2) scaling)
                tau_E = (env.H_H) * (0.0562) * (env.I_P^0.93) * (env.B_T^0.15) * (M^0.19) * ...
                    (env.R^1.97) * (env.epsilon^0.58) * (env.kappa^0.78) * ((P/1e6)^-0.69) * ...
                    (env.V^-0.69) * ((n_e_pre_step/1e19)^0.41);
                
                % Species-specific confinement times
                tau_Ei = tau_E * env.zeta_i;        % Ion energy confinement time
                tau_Ee = tau_E * env.zeta_e;        % Electron energy confinement time
                tau_D = tau_E * env.k_D;            % Deuterium confinement time
                tau_T = tau_E * env.k_T;            % Tritium confinement time
                tau_I = tau_E * env.k_I;            % Wall impurity confinement time
                tau_s = tau_E * env.k_s;            % Seeded impurity confinement time
                tau_alpha = tau_E * env.k_alpha;    % Alpha particle confinement time
    
                % Calculate recycling contributions
                factor = (((1 - env.f_ref * (1 - env.f_eff)) * env.R_eff) / ...
                    (1 - env.R_eff * (1 - env.f_eff)) - env.f_ref) * (n_D/tau_D + n_T/tau_T);
                
                S_D_R = 1/(1 - env.f_ref * (1 - env.f_eff)) * ...
                    (env.f_ref * n_D / tau_D + (1 - env.gamma_pfc) * factor);  % D recycling
                
                S_T_R = 1/(1 - env.f_ref * (1 - env.f_eff)) * ...
                    (env.f_ref * n_T / tau_T + env.gamma_pfc * factor);  % T recycling
    
                % Compute time derivatives of state variables
                n_alpha_dot = -(n_alpha / tau_alpha) + S_alpha;
                n_D_dot = -(n_D / tau_D) - S_alpha + S_D + env.f_eff * S_D_R;
                n_T_dot = -(n_T / tau_T) - S_alpha + S_T + env.f_eff * S_T_R;
                n_s_dot = -n_s / tau_s;
                
                n_I_dot = (-n_I / tau_I + env.S_I_inj + env.f_I_sp * n_tot_pre_step / tau_I + ...
                    env.f_I_sp * (3 * n_alpha_dot + 2 * (n_D_dot + n_T_dot) + (env.Z_s + 1)*n_s_dot)) / ...
                    (1 - env.f_I_sp * (1 + env.Z_I));
                
                Ei_dot = -(Ei/tau_Ei) + (env.phi_alpha * P_alpha) + P_ei + Paux_i;
                Ee_dot = -(Ee/tau_Ee) + ((1 - env.phi_alpha) * P_alpha) - P_ei - P_br + P_oh + Paux_e;
                
                % Euler integration step
                Ee = Ee + env.Ts * Ee_dot;
                Ei = Ei + env.Ts * Ei_dot;
                n_D = n_D + env.Ts * n_D_dot;
                n_T = n_T + env.Ts * n_T_dot;
                n_I = n_I + env.Ts * n_I_dot;
                n_alpha = n_alpha + env.Ts * n_alpha_dot;
                n_s = n_s + env.Ts * n_s_dot;
    
                % Apply lower bounds to prevent negative densities/energies
                Ee = clip(Ee, 1, Inf);
                Ei = clip(Ei, 1, Inf);
                n_D = clip(n_D, 1, Inf);
                n_T = clip(n_T, 1, Inf);
                n_I = clip(n_I, 1, Inf);
                n_alpha = clip(n_alpha, 1, Inf);
                n_s = clip(n_s, 1, Inf);
            end
            
            % Compute final state variables
            n_e_pre_step = n_D + n_T + 2*n_alpha + env.Z_I*n_I + env.Z_s*n_s;
            n_i = n_D + n_T + n_alpha + n_I + n_s;
            n = n_e_pre_step + n_i;
            gamma = n_T / (n_D + n_T);
            
            % Validate result is within physical bounds; retry if not
            if n < 1e20 || n > 2e20 || Ee < 1e5 || Ei < 1e5 || Ee > 2e5 || Ei > 2e5
                [Ee, Ei, n, gamma] = feedForward(env, env.stateSS(1), env.stateSS(2), ...
                    env.stateSS(3), env.stateSS(4), env.stateSS(5), env.stateSS(6), env.stateSS(7));
            else
                env.stateSS = [n_alpha; n_D; n_T; n_I; Ee; Ei; n_s];
            end
            
        end
        
        % -----------------------------------------------------------------
        % Step Function: Execute one time step with given action
        % -----------------------------------------------------------------
        function [Observation, Reward, IsDone] = step(env, Action)
            
            % Scale normalized actions to physical units
            F = Action .* [20e6; 20e6; 16.5e6; 16.5e6; 10e18; 10e18];

            % Compute heating power densities
            Paux_i = (env.eta_ic*F(1) + env.eta_nbi1*env.phi_nbi*F(3) + ...
                env.eta_nbi2*env.phi_nbi*F(4)) / env.V;  % Ion heating
            
            Paux_e = (env.eta_ec*F(2) + env.eta_nbi1*(1-env.phi_nbi)*F(3) + ...
                env.eta_nbi2*(1-env.phi_nbi)*F(4)) / env.V;  % Electron heating
            
            % Compute fueling rates
            S_D = env.eta_DT * (1 - env.gamma_DT) * F(6) + env.eta_D * F(5);  % Deuterium source
            S_T = env.eta_DT * env.gamma_DT * F(6);  % Tritium source
            
            % Extract current state variables
            n_alpha = env.State(1);     % Alpha particle density (m^-3)
            n_D = env.State(2);         % Deuterium density (m^-3)
            n_T = env.State(3);         % Tritium density (m^-3)
            n_I = env.State(4);         % Wall impurity density (m^-3)
            Ee = env.State(5);          % Electron energy density (J/m^3)
            Ei = env.State(6);          % Ion energy density (J/m^3)
            n_s = env.State(7);         % Seeded impurity density (m^-3)

            % Update target values with smooth transitions if enabled
            if env.rampOn == true
                if env.count < env.target_length + 1
                    % Still in initial target tracking period
                else
                    % Check if it's time to generate a new target
                    if mod(env.count, (env.target_length + env.target_transition_time + 1)) == 0 && ...
                            env.count/(env.target_length + 1 + env.target_transition_time) ~= 1 || ...
                            env.count/(env.target_length + 1) == 1
                        
                        % Generate new steady-state target
                        [target_Ee_bar, target_Ei_bar, target_n_bar, target_gamma_bar] = ...
                            feedForward(env, env.stateSS(1), env.stateSS(2), env.stateSS(3), ...
                            env.stateSS(4), env.stateSS(5), env.stateSS(6), env.stateSS(7));
    
                        % Randomly select transition type (linear or logarithmic)
                        env.use_logarithmic_transition = randi([0, 1]);
    
                        % Calculate step increments based on transition type
                        if env.use_logarithmic_transition == 1
                            % Logarithmic transition for smooth exponential changes
                            env.step_Ee = (log(target_Ee_bar) - log(env.Ee_bar)) / env.target_transition_time;
                            env.step_Ei = (log(target_Ei_bar) - log(env.Ei_bar)) / env.target_transition_time;
                            env.step_n = (log(target_n_bar) - log(env.n_bar)) / env.target_transition_time;
                            env.step_gamma = (log(target_gamma_bar) - log(env.gamma_bar)) / env.target_transition_time;
                        else
                            % Linear transition for uniform changes
                            env.step_Ee = (target_Ee_bar - env.Ee_bar) / env.target_transition_time;
                            env.step_Ei = (target_Ei_bar - env.Ei_bar) / env.target_transition_time;
                            env.step_n = (target_n_bar - env.n_bar) / env.target_transition_time;
                            env.step_gamma = (target_gamma_bar - env.gamma_bar) / env.target_transition_time;
                        end
    
                        env.transition_steps = env.target_transition_time;
                    end
    
                    % Apply smooth transition if in transition period
                    if env.transition_steps > 0
                        if env.use_logarithmic_transition == 1
                            % Update using exponential steps
                            env.Ee_bar = exp(log(env.Ee_bar) + env.step_Ee);
                            env.Ei_bar = exp(log(env.Ei_bar) + env.step_Ei);
                            env.n_bar = exp(log(env.n_bar) + env.step_n);
                            env.gamma_bar = exp(log(env.gamma_bar) + env.step_gamma);
                        else
                            % Update using linear steps
                            env.Ee_bar = env.Ee_bar + env.step_Ee;
                            env.Ei_bar = env.Ei_bar + env.step_Ei;
                            env.n_bar = env.n_bar + env.step_n;
                            env.gamma_bar = env.gamma_bar + env.step_gamma;
                        end
    
                        env.transition_steps = env.transition_steps - 1;
                    end
                end
            else
                % Step target updates (no smooth transitions)
                if mod(env.count, (env.target_length + 1)) == 0
                    [env.Ee_bar, env.Ei_bar, env.n_bar, env.gamma_bar] = ...
                        feedForward(env, env.stateSS(1), env.stateSS(2), env.stateSS(3), ...
                        env.stateSS(4), env.stateSS(5), env.stateSS(6), env.stateSS(7));
                end
            end

            % Compute current tritium fraction
            env.gamma = n_T / (n_D + n_T);

            % Calculate average ion mass number
            M = 3 * env.gamma + 2 * (1 - env.gamma);
            
            % Calculate particle densities
            n_e_pre_step = n_D + n_T + 2*n_alpha + env.Z_I*n_I + env.Z_s*n_s;
            n_i = n_D + n_T + n_alpha + n_I + n_s;
            n_tot_pre_step = n_e_pre_step + n_i;

            % Compute effective charge
            Z_eff = (n_D + n_T + 4*n_alpha + (env.Z_I^2)*n_I + (env.Z_s^2)*n_s) / n_e_pre_step;

            % Calculate temperatures from energy densities
            T_ion_J = ((2/3) * Ei / n_i) * (1 + env.gamma_t);
            T_ion_ev = (((2/3) * Ei / n_i) / env.e_charge) * (1 + env.gamma_t);
            T_ion_kev = T_ion_ev / 1000;
            
            T_e_J = ((2/3) * Ee / n_e_pre_step) * (1 + env.gamma_t);
            T_e_ev = (((2/3) * Ee / n_e_pre_step) / env.e_charge) * (1 + env.gamma_t);
            T_e_kev = T_e_ev / 1000;
            T_e_k = T_e_ev / env.ev_to_k;
            
            % Calculate Coulomb logarithm
            lamda_e = (1.24e7)*(T_e_k^1.5)/(Z_eff^2 * sqrt(n_e_pre_step))  / (1 + 3*env.gamma_t/2);
            lnAe = log(lamda_e);

            % Compute D-T fusion reactivity
            omega = T_ion_kev / (1 - (T_ion_kev * (env.C2 + T_ion_kev * (env.C4 + env.C6*T_ion_kev))) / ...
            (1 + T_ion_kev * (env.C3 + T_ion_kev * (env.C5 + env.C7*T_ion_kev))));
        
            zeta = (env.BG^2 / (4*omega))^(1/3);
            sigma_v = env.C1 * omega * sqrt(zeta / (env.mr*T_ion_kev^3)) * exp(-3*zeta) * 1e-6;
            S_alpha = env.gamma * (1 - env.gamma) * (n_D + n_T)^2 * sigma_v;
    
            % Calculate electron-ion collision time
            if n_s < 1e3
                tau_ei = (((2*pi)^0.5 * 3*pi * env.epsilon_0^2 * T_e_J^1.5) / ...
                    (env.e_charge^4 * env.m_e^0.5 * lnAe)) * ...
                    (env.m_D/n_D + env.m_T/n_T + env.m_alpha/(4*n_alpha) + env.m_I/(n_I*env.Z_I^2)) / ...
                    (1 + 3*env.gamma_t/2);
    
            elseif n_I < 1e3
                tau_ei = (((2*pi)^0.5 * 3*pi * env.epsilon_0^2 * T_e_J^1.5) / ...
                    (env.e_charge^4 * env.m_e^0.5 * lnAe)) * ...
                    (env.m_D/n_D + env.m_T/n_T + env.m_s/(n_s*env.Z_s^2) + env.m_alpha/(4*n_alpha)) / ...
                    (1 + 3*env.gamma_t/2);
                    
            elseif n_I < 1e3 && n_s < 1e3
                tau_ei = (((2*pi)^0.5 * 3*pi * env.epsilon_0^2 * T_e_J^1.5) / ...
                    (env.e_charge^4 * env.m_e^0.5 * lnAe)) * ...
                    (env.m_D/n_D + env.m_T/n_T + env.m_alpha/(4*n_alpha)) / ...
                    (1 + 3*env.gamma_t/2);
            else
                tau_ei = (((2*pi)^0.5 * 3*pi * env.epsilon_0^2 * T_e_J^1.5) / ...
                    (env.e_charge^4 * env.m_e^0.5 * lnAe)) * ...
                    (env.m_D/n_D + env.m_T/n_T + env.m_s/(n_s*env.Z_s^2) + ...
                    env.m_alpha/(4*n_alpha) + env.m_I/(n_I*env.Z_I^2)) / ...
                    (1 + 3*env.gamma_t/2);
            end
    
            % Calculate power sources and sinks
            P_alpha = (env.Q_alpha * env.e_charge) * S_alpha;
            P_ei = 1.5 * n_e_pre_step * ((T_e_J - T_ion_J) / tau_ei);
            P_oh = 2.8e-9 * Z_eff * (env.I_P*1e6)^2 / (env.a^4 * T_e_kev^1.5);
            P_br = (5.5e-37) * (n_D + n_T + 4*n_alpha + n_I*env.Z_I^2 + n_s*env.Z_s^2) * ...
                n_e_pre_step * sqrt(T_e_kev);
            
            P = max(Paux_i + Paux_e - P_br + P_alpha + P_oh, 1e-3);
            
            % Energy confinement time (IPB98(y,2) scaling)
            tau_E = (env.H_H) * (0.0562) * (env.I_P^0.93) * (env.B_T^0.15) * (M^0.19) * ...
                (env.R^1.97) * (env.epsilon^0.58) * (env.kappa^0.78) * ((P/1e6)^-0.69) * ...
                (env.V^-0.69) * ((n_e_pre_step/1e19)^0.41);
            
            % Species-specific confinement times
            tau_Ei = tau_E * env.zeta_i;
            tau_Ee = tau_E * env.zeta_e;
            tau_D = tau_E * env.k_D;
            tau_T = tau_E * env.k_T;
            tau_I = tau_E * env.k_I;
            tau_s = tau_E * env.k_s;
            tau_alpha = tau_E * env.k_alpha;
    
            % Calculate recycling contributions
            factor = (((1 - env.f_ref * (1 - env.f_eff)) * env.R_eff) / ...
                (1 - env.R_eff * (1 - env.f_eff)) - env.f_ref) * (n_D/tau_D + n_T/tau_T);
            
            S_D_R = 1/(1 - env.f_ref * (1 - env.f_eff)) * ...
                (env.f_ref * n_D / tau_D + (1 - env.gamma_pfc) * factor);
            
            S_T_R = 1/(1 - env.f_ref * (1 - env.f_eff)) * ...
                (env.f_ref * n_T / tau_T + env.gamma_pfc * factor);
    
            % Compute time derivatives of state variables
            n_alpha_dot = -(n_alpha / tau_alpha) + S_alpha;
            n_D_dot = -(n_D / tau_D) - S_alpha + S_D + env.f_eff * S_D_R;
            n_T_dot = -(n_T / tau_T) - S_alpha + S_T + env.f_eff * S_T_R;
            n_s_dot = -n_s / tau_s;
            
            n_I_dot = (-n_I / tau_I + env.S_I_inj + env.f_I_sp * n_tot_pre_step / tau_I + ...
                env.f_I_sp * (3 * n_alpha_dot + 2 * (n_D_dot + n_T_dot) + (env.Z_s + 1)*n_s_dot)) / ...
                (1 - env.f_I_sp * (1 + env.Z_I));
            
            Ei_dot = -(Ei/tau_Ei) + (env.phi_alpha * P_alpha) + P_ei + Paux_i;
            Ee_dot = -(Ee/tau_Ee) + ((1 - env.phi_alpha) * P_alpha) - P_ei - P_br + P_oh + Paux_e;
    
            % Euler integration to update state
            env.State = env.State + env.Ts .* [n_alpha_dot; n_D_dot; n_T_dot; n_I_dot; Ee_dot; Ei_dot; n_s_dot];
           
            % Apply lower bounds to state variables
            env.State(1:7) = clip(env.State(1:7), 1, Inf);
    
            % Update total density and tritium fraction
            n_e_step = env.State(2) + env.State(3) + 2*env.State(1) + env.Z_I*env.State(4) + env.Z_s*env.State(5);
            env.n_tot = env.State(2) + env.State(3) + env.State(1) + env.State(4) + n_e_step + env.State(5);
            env.gamma = n_T / (n_T + n_D);
            
            % Compute reward based on tracking errors
            Ei_error = abs(env.State(6) - env.Ei_bar) / env.Ei_bar;
            Ee_error = abs(env.State(5) - env.Ee_bar) / env.Ee_bar;
            n_step_error = abs(env.n_tot - env.n_bar) / env.n_bar;
            gamma_error = abs(env.gamma - env.gamma_bar) / env.gamma_bar;
    
            % Multi-scale Gaussian reward function
            sigma1 = 0.25;
            sigma2 = 0.05;
            sigma3 = 0.005;
            
            Reward = (1/3) * (exp(-(Ei_error^2)/(2*sigma1^2)) + exp(-(Ei_error^2)/(2*sigma2^2)) + ...
                exp(-(Ei_error^2)/(2*sigma3^2)));
            Reward = Reward + (1/3) * (exp(-(Ee_error^2)/(2*sigma1^2)) + exp(-(Ee_error^2)/(2*sigma2^2)) + ...
                exp(-(Ee_error^2)/(2*sigma3^2)));
            Reward = Reward + (1/3) * (exp(-(n_step_error^2)/(2*sigma1^2)) + exp(-(n_step_error^2)/(2*sigma2^2)) + ...
                exp(-(n_step_error^2)/(2*sigma3^2)));
            Reward = Reward + (1/3) * (exp(-(gamma_error^2)/(2*sigma1^2)) + exp(-(gamma_error^2)/(2*sigma2^2)) + ...
                exp(-(gamma_error^2)/(2*sigma3^2)));
            
            % Episode never terminates (continuous operation)
            IsDone = false;
            
            % Construct normalized observation vector
            Observation = [
                env.State(5) / 1e5;     % Electron energy (normalized)
                env.State(6) / 1e5;     % Ion energy (normalized)
                env.State(3) / 1e20;    % Tritium density (normalized)
                env.State(2) / 1e20;    % Deuterium density (normalized)
                env.State(4) / 1e20;    % Wall impurity density (normalized)
                env.State(1) / 1e20;    % Alpha density (normalized)
                env.n_tot / 1e20;       % Total density (normalized)
                env.gamma;              % Tritium fraction
                env.n_bar / 1e20;       % Target total density (normalized)
                env.gamma_bar;          % Target tritium fraction
                env.Ee_bar / 1e5;       % Target electron energy (normalized)
                env.Ei_bar / 1e5        % Target ion energy (normalized)
            ];
    
            % Increment step counter and update plots
            env.count = env.count + 1;
            envUpdatedCallback(env, Action);
            
        end
    
        % -----------------------------------------------------------------
        % Reset Function: Initialize environment to starting conditions
        % -----------------------------------------------------------------
        function InitialObservation = reset(env)
            
            % Set initial particle densities (m^-3)
            n_alpha_i = 2e18;       % Alpha particles
            n_D_i = 3e19;           % Deuterium
            n_T_i = 3e19;           % Tritium
            n_I_i = 1e18;           % Wall impurity
            n_s_i = 0;              % Seeded impurity
    
            % Calculate initial tritium fraction
            env.gamma = n_T_i / (n_T_i + n_D_i);
    
            % Calculate initial electron density
            n_e = n_D_i + n_T_i + 2*n_alpha_i + env.Z_I*n_I_i + env.Z_s*n_s_i;
            
            % Calculate initial total particle density
            env.n_tot = n_T_i + n_D_i + n_alpha_i + n_I_i + n_e + n_s_i;
                    
            % Set initial energy densities (J/m^3)
            Ee_i = 1.8e5;           % Electron energy
            Ei_i = 1.5e5;           % Ion energy
    
            % Initialize state vector
            env.State = [n_alpha_i; n_D_i; n_T_i; n_I_i; Ee_i; Ei_i; n_s_i];
            env.statei = env.State;
            
            % Generate initial target conditions
            [env.Ee_bar, env.Ei_bar, env.n_bar, env.gamma_bar] = ...
                feedForward(env, env.statei(1), env.statei(2), env.statei(3), ...
                env.statei(4), env.statei(5), env.statei(6), env.statei(7));
           
            % Construct initial observation
            InitialObservation = [
                Ee_i / 1e5;
                Ei_i / 1e5;
                n_T_i / 1e20;
                n_D_i / 1e20;
                n_I_i / 1e20;
                n_alpha_i / 1e20;
                env.n_tot / 1e20;
                env.gamma;
                env.n_bar / 1e20;
                env.gamma_bar;
                env.Ee_bar / 1e5;
                env.Ei_bar / 1e5
            ];
            
            % Reset step counter
            env.count = 1;
    
            % Initialize plots
            envUpdatedCallback(env, [0; 0; 0; 0; 0; 0]);
             
        end
        
        %% ================================================================
        %  PLOTTING METHODS
        %  ================================================================
        
        % -----------------------------------------------------------------
        % Initialize electron energy plot
        % -----------------------------------------------------------------
        function plotEe(env)
            env.Ee_figure = figure("HandleVisibility", "off", "Name", "Ee");
            h = gca(env.Ee_figure);
            hold(h, 'on');
            
            h.XLim = [0, env.TT];
            h.YLim = [1.5e5, 1.8e5];
            h.FontSize = 30;
            
            legend(h, 'Orientation', 'vertical', 'Location', 'northeast', 'FontSize', 25);
            xlabel(h, 'Time (s)', 'FontSize', 40);
            ylabel(h, 'Electron Energy ($\mathrm{J/m^3}$)', 'Interpreter', 'latex', 'FontSize', 40);
            
            grid(h, 'on');
            grid(h, 'minor');
            pbaspect(h, [2.5 1 1]);
        end
        
        % -----------------------------------------------------------------
        % Initialize ion energy plot
        % -----------------------------------------------------------------
        function plotEi(env)
            env.Ei_figure = figure("HandleVisibility", "off", "Name", "Ei");
            h = gca(env.Ei_figure);
            hold(h, 'on');
            
            h.XLim = [0, env.TT];
            h.YLim = [1.2e5, 1.65e5];
            h.FontSize = 30;
            
            legend(h, 'Orientation', 'vertical', 'Location', 'northeast', 'FontSize', 25);
            xlabel(h, 'Time (s)', 'FontSize', 40);
            ylabel(h, 'Ion Energy ($\mathrm{J/m^3}$)', 'Interpreter', 'latex', 'FontSize', 40);
            
            grid(h, 'on');
            grid(h, 'minor');
            pbaspect(h, [2.5 1 1]);
        end
        
        % -----------------------------------------------------------------
        % Initialize total particle density plot
        % -----------------------------------------------------------------
        function plotN(env)
            env.n_tot_figure = figure("HandleVisibility", "off", "Name", "Particle Density");
            h = gca(env.n_tot_figure);
            hold(h, 'on');
            
            h.XLim = [0, env.TT];
            h.YLim = [1.3e20, 2.1e20];
            h.FontSize = 30;
            
            legend(h, 'Orientation', 'vertical', 'Location', 'northeast', 'FontSize', 25);
            xlabel(h, 'Time (s)', 'FontSize', 40);
            ylabel(h, 'Particle Density ($\mathrm{m^{-3}}$)', 'Interpreter', 'latex', 'FontSize', 40);
            
            grid(h, 'on');
            grid(h, 'minor');
            pbaspect(h, [2.5 1 1]);
        end
        
        % -----------------------------------------------------------------
        % Initialize auxiliary heating plot
        % -----------------------------------------------------------------
        function plotHeating(env)
            env.heatingFigure = figure("HandleVisibility", "off", "Name", "Auxiliary Heating");
            h = gca(env.heatingFigure);
            hold(h, 'on');
            
            h.XLim = [0, env.TT];
            h.YLim = [0, 20];
            h.FontSize = 30;
            
            legend(h, 'Orientation', 'vertical', 'Location', 'northeast', 'FontSize', 25);
            xlabel(h, 'Time (s)', 'FontSize', 40);
            ylabel(h, 'Heating (MW)', 'Interpreter', 'latex', 'FontSize', 40);
            
            grid(h, 'on');
            pbaspect(h, [2.5 1 1]);
        end
        
        % -----------------------------------------------------------------
        % Initialize deuterium density plot
        % -----------------------------------------------------------------
        function plotD(env)
            env.n_D_figure = figure("HandleVisibility", "off", "Name", "Deuterium Density");
            h = gca(env.n_D_figure);
            h.XLim = [0, env.TT];
            
            title(h, 'Deuterium Density vs Time');
            xlabel(h, 'Time (ds)');
            ylabel(h, 'Deuterium Particle Density (1/m^3)');
            
            grid(h, 'on');
            hold(h, 'on');
        end
        
        % -----------------------------------------------------------------
        % Initialize tritium density plot
        % -----------------------------------------------------------------
        function plotT(env)
            env.n_T_figure = figure("HandleVisibility", "off", "Name", "Tritium Density");
            h = gca(env.n_T_figure);
            h.XLim = [0, env.TT];
            
            title(h, 'Tritium Density vs Time');
            xlabel(h, 'Time (ds)');
            ylabel(h, 'Tritium Particle Density (1/m^3)');
            
            grid(h, 'on');
            hold(h, 'on');
        end
        
        % -----------------------------------------------------------------
        % Initialize wall impurity density plot
        % -----------------------------------------------------------------
        function plotI(env)
            env.n_I_figure = figure("HandleVisibility", "off", "Name", "Wall Impurity Density");
            h = gca(env.n_I_figure);
            h.XLim = [0, env.TT];
            
            title(h, 'Wall Impurity Density');
            xlabel(h, 'Time (ds)');
            ylabel(h, 'Wall Impurity Particle Density (1/m^3)');
            
            grid(h, 'on');
            hold(h, 'on');
        end
        
        % -----------------------------------------------------------------
        % Initialize alpha particle density plot
        % -----------------------------------------------------------------
        function plotAlpha(env)
            env.n_alpha_figure = figure("HandleVisibility", "off", "Name", "Alpha Particle Density");
            h = gca(env.n_alpha_figure);
            h.XLim = [0, env.TT];
            
            title(h, 'Alpha Particle Density vs Time');
            xlabel(h, 'Time (s)');
            ylabel(h, 'Particle Density (1/m^3)');
            
            grid(h, 'on');
            hold(h, 'on');
        end
        
        % -----------------------------------------------------------------
        % Initialize tritium fraction plot
        % -----------------------------------------------------------------
        function plotGamma(env)
            env.gamma_figure = figure("HandleVisibility", "off", "Name", "Tritium Fraction");
            h = gca(env.gamma_figure);
            hold(h, 'on');
            
            h.XLim = [0, env.TT];
            h.YLim = [0.45, 0.55];
            h.FontSize = 30;
            
            legend(h, 'Orientation', 'vertical', 'Location', 'northeast', 'FontSize', 25);
            xlabel(h, 'Time (s)', 'FontSize', 40);
            ylabel(h, 'Tritium Fraction', 'FontSize', 40);
            
            grid(h, 'minor');
            pbaspect(h, [2.5 1 1]);
        end
        
        % -----------------------------------------------------------------
        % Initialize fueling rates plot
        % -----------------------------------------------------------------
        function plotFueling(env)
            env.fuelingFigure = figure("HandleVisibility", "off", "Name", "Fueling");
            h = gca(env.fuelingFigure);
            hold(h, 'on');
            
            h.XLim = [0, env.TT];
            h.YLim = [0, 1e19];
            h.FontSize = 30;
            
            legend(h, 'Orientation', 'vertical', 'Location', 'northeast', 'FontSize', 25);
            xlabel(h, 'Time (s)', 'FontSize', 40);
            ylabel(h, 'Fueling ($\mathrm{m^{-3}s^{-1}}$)', 'Interpreter', 'latex', 'FontSize', 40);
            
            grid(h, 'on');
            pbaspect(h, [2.5 1 1]);
        end
        
    end
    
    methods(Access = protected)
        
        % -----------------------------------------------------------------
        % Callback: Update all active plots with current state
        % -----------------------------------------------------------------
        function envUpdatedCallback(env, Action)
            
            % Update electron energy plot
            if ~isempty(env.Ee_figure) && isvalid(env.Ee_figure)
                h = gca(env.Ee_figure);
                Ee_line = findobj(h, "Tag", "Ee_line");
                Ee_bar_line = findobj(h, "Tag", "Ee_bar_line");
    
                if any([isempty(Ee_line), ~isvalid(Ee_line)])
                    Ee_line = animatedline(h, [], [], "Tag", "Ee_line", "Color", "k", ...
                        "LineWidth", 5, "Linestyle", "-", "DisplayName", "Actual");
                end
    
                if any([isempty(Ee_bar_line), ~isvalid(Ee_bar_line)])
                    Ee_bar_line = animatedline(h, [], [], "Tag", "Ee_bar_line", "Color", "r", ...
                        "LineWidth", 4.9, "Linestyle", "--", "DisplayName", "Reference");
                end
    
                if env.count == 1
                    clearpoints(Ee_line);
                    clearpoints(Ee_bar_line);
                end
                
                addpoints(Ee_line, env.count*env.Ts, env.State(5));
                addpoints(Ee_bar_line, env.count*env.Ts, env.Ee_bar);
                drawnow limitrate nocallbacks
            end
            
            % Update ion energy plot
            if ~isempty(env.Ei_figure) && isvalid(env.Ei_figure)
                h = gca(env.Ei_figure);
                Ei_line = findobj(h, "Tag", "Ei_line");
                Ei_bar_line = findobj(h, "Tag", "Ei_bar_line");
    
                if any([isempty(Ei_line), ~isvalid(Ei_line)])
                    Ei_line = animatedline(h, [], [], "Tag", "Ei_line", "Color", "k", ...
                        "LineWidth", 5, "Linestyle", "-", "DisplayName", "Actual");
                end
                
                if any([isempty(Ei_bar_line), ~isvalid(Ei_bar_line)])
                    Ei_bar_line = animatedline(h, [], [], "Tag", "Ei_bar_line", "Color", "r", ...
                        "LineWidth", 4.9, "Linestyle", "--", "DisplayName", "Reference");
                end
                
                if env.count == 1
                    clearpoints(Ei_line);
                    clearpoints(Ei_bar_line);
                end
                
                addpoints(Ei_line, env.count*env.Ts, env.State(6));
                addpoints(Ei_bar_line, env.count*env.Ts, env.Ei_bar);
                drawnow limitrate nocallbacks
            end
            
            % Update total density plot
            if ~isempty(env.n_tot_figure) && isvalid(env.n_tot_figure)
                h = gca(env.n_tot_figure);
                n_line = findobj(h, "Tag", "n_line");
                n_bar_line = findobj(h, "Tag", "n_bar_line");
    
                if any([isempty(n_line), ~isvalid(n_line)])
                    n_line = animatedline(h, [], [], "Tag", "n_line", "Color", "k", ...
                        "LineWidth", 5, "Linestyle", "-", "DisplayName", "Actual");
                end
                
                if any([isempty(n_bar_line), ~isvalid(n_bar_line)])
                    n_bar_line = animatedline(h, [], [], "Tag", "n_bar_line", "Color", "r", ...
                        "LineWidth", 4.9, "Linestyle", "--", "DisplayName", "Reference");
                end
                
                if env.count == 1
                    clearpoints(n_line);
                    clearpoints(n_bar_line);
                end
                
                addpoints(n_line, env.count*env.Ts, env.n_tot);
                addpoints(n_bar_line, env.count*env.Ts, env.n_bar);
                drawnow limitrate nocallbacks
            end
            
            % Update deuterium density plot
            if ~isempty(env.n_D_figure) && isvalid(env.n_D_figure)
                h = gca(env.n_D_figure);
                n_D_line = findobj(h, "Tag", "n_D_line");
    
                if any([isempty(n_D_line), ~isvalid(n_D_line)])
                    n_D_line = animatedline(h, [], [], "Tag", "n_D_line", "Color", "black", "LineWidth", 9);
                end
                
                if env.count == 1
                    clearpoints(n_D_line);
                end
                
                addpoints(n_D_line, env.count, env.State(2));
                drawnow limitrate nocallbacks
            end
            
            % Update tritium density plot
            if ~isempty(env.n_T_figure) && isvalid(env.n_T_figure)
                h = gca(env.n_T_figure);
                n_T_line = findobj(h, "Tag", "n_T_line");
    
                if any([isempty(n_T_line), ~isvalid(n_T_line)])
                    n_T_line = animatedline(h, [], [], "Tag", "n_T_line", "Color", "black", "LineWidth", 9);
                end
                
                if env.count == 1
                    clearpoints(n_T_line);
                end
                
                addpoints(n_T_line, env.count, env.State(3));
                drawnow limitrate nocallbacks
            end
            
            % Update wall impurity density plot
            if ~isempty(env.n_I_figure) && isvalid(env.n_I_figure)
                h = gca(env.n_I_figure);
                n_I_line = findobj(h, "Tag", "n_I_line");
    
                if any([isempty(n_I_line), ~isvalid(n_I_line)])
                    n_I_line = animatedline(h, [], [], "Tag", "n_I_line", "Color", "black", "LineWidth", 9);
                end
                
                if env.count == 1
                    clearpoints(n_I_line);
                end
                
                addpoints(n_I_line, env.count, env.State(4));
                drawnow limitrate nocallbacks
            end
            
            % Update alpha particle density plot
            if ~isempty(env.n_alpha_figure) && isvalid(env.n_alpha_figure)
                h = gca(env.n_alpha_figure);
                n_alpha_line = findobj(h, "Tag", "n_alpha_line");
    
                if any([isempty(n_alpha_line), ~isvalid(n_alpha_line)])
                    n_alpha_line = animatedline(h, [], [], "Tag", "n_alpha_line", "Color", "black", "LineWidth", 9);
                end
                
                if env.count == 1
                    clearpoints(n_alpha_line);
                end
                
                addpoints(n_alpha_line, env.count, env.State(1));
                drawnow limitrate nocallbacks
            end
            
            % Update tritium fraction plot
            if ~isempty(env.gamma_figure)&&isvalid(env.gamma_figure)
                h = gca(env.gamma_figure);
                gamma_line = findobj(h,"Tag","gamma_line");
                gamma_bar_line = findobj(h,"Tag","gamma_bar_line");

                if any([isempty(gamma_line),~isvalid(gamma_line)])
                    gamma_line = animatedline(h,[],[],"Tag","gamma_line", "Color","k", "LineWidth",5,"Linestyle","-",DisplayName= "Actual");
                end
                if any([isempty(gamma_bar_line),~isvalid(gamma_bar_line)])
                    gamma_bar_line = animatedline(h,[],[],"Tag","gamma_bar_line", "Color","r", "LineWidth",4.9,"Linestyle","--",DisplayName= "Reference");
                end
                if env.count ==1
                    clearpoints(gamma_line);
                    clearpoints(gamma_bar_line);
                end
                addpoints(gamma_line,env.count*env.Ts,env.gamma)
                addpoints(gamma_bar_line,env.count*env.Ts,env.gamma_bar)
                drawnow limitrate nocallbacks
            end
            
            % Update fueling rates plot
            if ~isempty(env.fuelingFigure) && isvalid(env.fuelingFigure)
                h = gca(env.fuelingFigure);
                D_line = findobj(h, "Tag", "D_line");
                DT_line = findobj(h, "Tag", "DT_line");
    
                if any([isempty(D_line), ~isvalid(D_line)])
                    D_line = animatedline(h, [], [], "Tag", "D_line", "Color", "blue", ...
                        "LineWidth", 5, "Linestyle", "-", "DisplayName", "Deuterium Injection");
                end
                
                if any([isempty(DT_line), ~isvalid(DT_line)])
                    DT_line = animatedline(h, [], [], "Tag", "DT_line", "Color", "black", ...
                        "LineWidth", 5, "Linestyle", "--", "DisplayName", "Deuterium-Tritium Injection");
                end
                
                if env.count == 1
                    clearpoints(D_line);
                    clearpoints(DT_line);
                end
                
                addpoints(D_line, env.count*env.Ts, Action(5)*10e18);
                addpoints(DT_line, env.count*env.Ts, Action(6)*10e18);
                drawnow limitrate nocallbacks
            end
            
            % Update heating powers plot
            if ~isempty(env.heatingFigure) && isvalid(env.heatingFigure)
                h = gca(env.heatingFigure);
                ECRH_line = findobj(h, "Tag", "ECRH_line");
                ICRH_line = findobj(h, "Tag", "ICRH_line");
                NBI1_line = findobj(h, "Tag", "NBI1_line");
                NBI2_line = findobj(h, "Tag", "NBI2_line");
    
                if any([isempty(ECRH_line), ~isvalid(ECRH_line)])
                    ECRH_line = animatedline(h, [], [], "Tag", "ECRH_line", "Color", "black", ...
                        "LineWidth", 5, "Linestyle", "--", "DisplayName", "ECRH");
                end
                
                if any([isempty(ICRH_line), ~isvalid(ICRH_line)])
                    ICRH_line = animatedline(h, [], [], "Tag", "ICRH_line", "Color", "blue", ...
                        "LineWidth", 5, "Linestyle", "-", "DisplayName", "ICRH");
                end
                
                if any([isempty(NBI1_line), ~isvalid(NBI1_line)])
                    NBI1_line = animatedline(h, [], [], "Tag", "NBI1_line", "Color", "red", ...
                        "LineWidth", 5, "Linestyle", "--", "DisplayName", "NBI1");
                end
                
                if any([isempty(NBI2_line), ~isvalid(NBI2_line)])
                    NBI2_line = animatedline(h, [], [], "Tag", "NBI2_line", "Color", "green", ...
                        "LineWidth", 5, "Linestyle", "-", "DisplayName", "NBI2");
                end
                
                if env.count == 1
                    clearpoints(ECRH_line);
                    clearpoints(ICRH_line);
                    clearpoints(NBI1_line);
                    clearpoints(NBI2_line);
                end
                
                addpoints(ECRH_line, env.count*env.Ts, Action(1)*20);
                addpoints(ICRH_line, env.count*env.Ts, Action(2)*20);
                addpoints(NBI1_line, env.count*env.Ts, Action(3)*16.5);
                addpoints(NBI2_line, env.count*env.Ts, Action(4)*16.5);
                drawnow limitrate nocallbacks
            end
            
        end
    end
end