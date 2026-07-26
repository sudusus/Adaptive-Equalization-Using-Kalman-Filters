% ------------------------------------------------------------------------
% Authors: Aaron B., James L. Breithaupt, Matthew F. Milano
% Date: 7/26/26
% Course: EE 553 - Topics in Digital Signal Processing
% Assignment: Midterm Project 2
% Topic: Kalman Filter Applications in Adaptive Equalization for High-Speed Channels
% Topic Focus: Restoring optical signals that undergo distortion effects
%              caused by fiber optics using adaptive equalization
%              and Kalman Filtering.
% ------------------------------------------------------------------------

clearvars;
clc;
close all;

%% System parameters
N = 5000;                 % Number of transmitted symbols
SNR = 25;                 % Channel SNR in dB
h = [0.2 0.9 0.4];        % Dispersive ISI channel impulse response
L = 8;                    % Number of adaptive equalizer taps
delay = 4;                % Equalizer decision delay

%% Binary Phase Shift Keying (BPSK) signal generation
% bit 0 -> -1
% bit 1 -> +1
bits = randi([0 1], N, 1);
x = 2*bits - 1;

% Channel model (Fiber channel with inter-symbol interference (ISI))
channel_output = filter(h, 1, x);

% Additive White Gaussian Noise (AWGN)
r = awgn(channel_output, SNR, 'measured');

%% Estimate channel noise variance
signalPower = mean(channel_output.^2);
noiseVariance = signalPower / (10^(SNR/10));

%% Kalman equalizer initialization

% State vector:
w = zeros(L, 1);

% State transition matrix
% The equalizer coefficients are assumed to change slowly.
F = eye(L);

% Initial state-estimation covariance
P = 10 * eye(L);

% Process-noise covariance
% Increasing this allows the equalizer to track faster channel changes.
Q = 1e-6 * eye(L);

% Measurement-noise covariance
R = max(noiseVariance, 1e-8);

% Identity matrix used in covariance update
I_L = eye(L);

%% Storage variables
y = zeros(N, 1);          % Equalizer output
e = zeros(N, 1);          % Kalman innovation/error
w_values = zeros(L, N);   % Equalizer coefficient history
P_trace = zeros(N, 1);    % Total coefficient uncertainty

%% Kalman adaptive equalization
for n = max(L, delay + 1):N

    % Most recent received samples
    u = r(n:-1:n-L+1);

    % Measurement matrix
    %
    % d(n) = H*w(n) + measurement noise
    H = u.';

    %% Prediction step

    % Predict equalizer coefficients
    w_predicted = F * w;

    % Predict coefficient covariance
    P_predicted = F * P * F.' + Q;

    %% Equalizer output

    % Output using predicted equalizer coefficients
    y(n) = H * w_predicted;

    % Known desired BPSK symbol
    d = x(n-delay);

    % Innovation or prediction error
    e(n) = d - y(n);

    %% Measurement update

    % Innovation covariance
    innovationVariance = H * P_predicted * H.' + R;

    % Kalman gain
    K = (P_predicted * H.') / innovationVariance;

    % Update equalizer tap coefficients
    w = w_predicted + K * e(n);

    % Covariance update
    covarianceUpdate = I_L - K * H;

    P = covarianceUpdate * P_predicted * covarianceUpdate.' ...
        + K * R * K.';

    % Remove small numerical asymmetry
    P = (P + P.') / 2;

    % Store results
    w_values(:, n) = w;
    P_trace(n) = trace(P);
end

%% Symbol detection and analysis

% Decisions before equalization
received = ones(size(r));
received(r < 0) = -1;

% Decisions after equalization
detected = ones(size(y));
detected(y < 0) = -1;


% BER calculation
start_idx = 1000;      % Ignore adaptation period

% BER before equalization
numErr_Before = sum(received(start_idx:end) ~= x(start_idx:end));
BER_Before = numErr_Before/length(received(start_idx:end));

% BER after equalization
detectedComparison = detected(start_idx:end);
originalAfter = x(start_idx-delay:end-delay);

numErr_After = sum(detectedComparison ~= originalAfter);
BER_After = numErr_After / length(detectedComparison);

fprintf('Bit Errors Before = %d\n', numErr_Before);
fprintf('BER Before Equalization = %.6f\n', BER_Before);

fprintf('Bit Errors After = %d\n', numErr_After);
fprintf('BER After Equalization = %.6f\n', BER_After);

fprintf('\nFinal equalizer coefficients:\n');
disp(w.');

%% Plotting
% Plot 1: Original and received signals
figure(1);

subplot(2,1,1);
plot(x(1:100));
title('Original BPSK Signal');
xlabel('Symbol');
ylabel('Amplitude');
ylim([-1.5 1.5]);
grid on;

subplot(2,1,2);
plot(r(1:100));
title('Received Signal After Dispersive Channel');
xlabel('Symbol');
ylabel('Amplitude');
grid on;

% Plot 2: Kalman convergence
mse_smooth = movmean(e.^2, 100);

figure(2);
plot(10*log10(mse_smooth + eps));
title('Kalman Equalizer Convergence');
xlabel('Iteration');
ylabel('Smoothed Error Power (dB)');
grid on;

% Plot 3: Equalizer coefficients over time
figure(3);
plot(w_values.');

title('Kalman Equalizer Coefficient Evolution');
xlabel('Iteration');
ylabel('Coefficient Value');
grid on;

legend('Tap 1', 'Tap 2', 'Tap 3', 'Tap 4', ...
       'Tap 5', 'Tap 6', 'Tap 7', 'Tap 8', ...
       'Location', 'best');

% Plot 4: Final equalizer coefficients
figure(4);
stem(1:L, w, 'filled');

title('Final Kalman Equalizer Coefficients');
xlabel('Tap Number');
ylabel('Coefficient Value');
grid on;

% Plot 5: Received and equalized signals
figure(5);

subplot(2,1,1);
plot(r(1:200));
title('Received Signal');
xlabel('Symbol');
ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot(y(1:200));
title('Kalman Equalized Signal');
xlabel('Symbol');
ylabel('Amplitude');
grid on;

% Plot 6: Received signal constellation
figure(6);
scatter(real(r(start_idx:end)), ...
        zeros(length(r(start_idx:end)),1), '.');

title('Received BPSK Constellation');
xlabel('In-Phase');
ylabel('Quadrature');
grid on;

% Plot 7: Equalized signal constellation
figure(7);
scatter(real(y(start_idx:end)), ...
        zeros(length(y(start_idx:end)),1), '.');

title('Kalman Equalized BPSK Constellation');
xlabel('In-Phase');
ylabel('Quadrature');
grid on;

% Plot 8: Eye diagram before equalization
eyediagram(r(start_idx:end), 2);
title('Eye Diagram Before Equalization');

% Plot 9: Eye diagram after equalization
eyediagram(y(start_idx:end), 2);
title('Eye Diagram After Kalman Equalization');

% Plot 10: Coefficient uncertainty
figure(10);
semilogy(P_trace(max(L, delay+1):end) + eps);

title('Kalman Coefficient Uncertainty');
xlabel('Iteration');
ylabel('Trace of P');
grid on;