  %% S003 RRB analysis -- FINAL MODEL + Cook's Distance / Leverage Diagnostics
%  
%     (1) The final stepwise-selected model (caretaker + feed + mood + choice),
%         n=17 (session 14 excluded), with cluster-robust SE.
%     (2) Cook's distance & leverage for the THEORY model (routine+feed+choice)
%         -- reproduces the Session-19 high-leverage finding.
%     (3) Cook's distance & leverage for the FINAL model -- shows influence
%         is well-distributed (no single dominant session).
%
%   Cook's distance here matches statsmodels' GLMInfluence.cooks_distance
%   under a fixed-dispersion (scale=1) negative binomial family:
%       D_i = pearson_resid_i^2 * h_i / (k * (1-h_i)^2)
%   where h_i is the IRLS-weighted hat-matrix diagonal at convergence and
%   k is the number of estimated parameters (including intercept).

clear; clc;

%% 1. Load data ---
filename = "path/to/yout/data.xlsx";
T = readtable(filename, "Sheet", "Coded Dataset v2");

usable = ismember(T.outcome_status, {'observed','no_RRB_zero'});
T = T(usable, :);

outcomes = ["obs1", "obs2"];

%% 2. Theory model (routine + feed + choice), n=17, excl. Session 14 ---
excl14 = T.session ~= 14; %& (T.session ~= 18) & (T.session ~= 19);
Te = T(excl14, :);
X_theory = [Te.self_routine_disruption, Te.feed_disruption, Te.choice_C];
names_theory = {'Intercept','self_routine_disruption','feed_disruption','choice_C'};


fprintf(' THEORY MODEL (n=17) -- Cook''s distance & leverage\n');


for outcome = outcomes
    y = Te.(outcome);
    keep = ~isnan(y);
    Xk = X_theory(keep,:); yk = y(keep); sess_k = Te.session(keep);

    [beta, mu, ~] = nbglm_fit(Xk, yk, 1.0);
    se_robust = cluster_robust_se(Xk, yk, mu, 1.0, Te.study_week(keep));
    print_table(outcome + " -- theory model (n=" + sum(keep) + ")", names_theory, beta, se_robust);

    [cooksD, lev] = cooks_distance_nb(Xk, yk, mu, 1.0);
    print_influence(outcome + " -- theory model influence", sess_k, cooksD, lev);
end

%% 3. FINAL MODEL: caretaker + feed + mood + choice, n=17 -----------------
X_final = [Te.caretaker_disruption, Te.feed_disruption, Te.mood_1to5, Te.choice_C]; %, Te.sleep_disruption
names_final = {'Intercept','caretaker_disruption','feed_disruption','mood_1to5','choice_C'}; %,'sleep_disruption'

fprintf('\n============================================================\n');
fprintf(' FINAL MODEL (n=17, excl. Session 14 only)\n');
fprintf(' caretaker_disruption + feed_disruption + mood_1to5 + choice_C\n');
fprintf('============================================================\n');

final_results = struct();
for outcome = outcomes
    y = Te.(outcome);
    keep = ~isnan(y);
    Xk = X_final(keep,:); yk = y(keep); sess_k = Te.session(keep);

    [beta, mu, ~] = nbglm_fit(Xk, yk, 1.0);
    se_robust = cluster_robust_se(Xk, yk, mu, 1.0, Te.study_week(keep));
    print_table(outcome + " -- FINAL model (n=" + sum(keep) + ")", names_final, beta, se_robust);

    [cooksD, lev] = cooks_distance_nb(Xk, yk, mu, 1.0);
    print_influence(outcome + " -- FINAL model influence (should be well-distributed)", ...
                     sess_k, cooksD, lev);

    final_results.(outcome) = struct('beta', beta, 'se', se_robust, ...
                                      'cooksD', cooksD, 'leverage', lev, 'session', sess_k);
end


%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================

function [beta, mu, W] = nbglm_fit(X, y, alpha)
% Fixed-dispersion Negative Binomial GLM, log link, fit via IRLS.
% Variance function: V(mu) = mu + alpha*mu^2   (alpha fixed, not estimated)
% Matches statsmodels' sm.families.NegativeBinomial(alpha=alpha).
    X = [ones(size(X,1),1), X];
    y = y(:);
    n = length(y);
    beta = zeros(size(X,2),1);
    mu = mean(y) * ones(n,1);
    eta = log(mu);
    beta(1) = log(mean(y));

    for iter = 1:100
        W = mu ./ (1 + alpha*mu);
        z = eta + (y - mu) ./ mu;
        Wd = diag(W);
        beta_new = (X' * Wd * X) \ (X' * Wd * z);
        eta = X * beta_new;
        mu = exp(eta);
        mu = max(mu, 1e-8);
        if max(abs(beta_new - beta)) < 1e-8
            beta = beta_new; break
        end
        beta = beta_new;
    end
end

function se = cluster_robust_se(X, y, mu, alpha, groups)
% Sandwich (cluster-robust) SE, clustered on `groups`, small-sample
% correction c = G/(G-1) * (N-1)/(N-K) (Stata/statsmodels-style).
    X = [ones(size(X,1),1), X];
    y = y(:); mu = mu(:);
    N = length(y); K = size(X,2);
    W = mu ./ (1 + alpha*mu);
    bread = inv(X' * diag(W) * X);

    ug = unique(groups);
    G = length(ug);
    meat = zeros(size(X,2));
    resid_score = (y - mu) ./ (1 + alpha*mu);
    for g = 1:G
        idx = groups == ug(g);
        sg = X(idx,:)' * resid_score(idx);
        meat = meat + sg * sg';
    end
    c = (G/(G-1)) * ((N-1)/(N-K));
    Vsand = c * (bread * meat * bread);
    se = sqrt(diag(Vsand));
end

function [cooksD, leverage] = cooks_distance_nb(X, y, mu, alpha)
% Cook's distance and leverage (hat diagonal) for a fixed-dispersion NB GLM,
% matching statsmodels' GLMInfluence.cooks_distance with scale=1 AND
% observed=True (statsmodels' default), i.e. the OBSERVED Hessian weight,
% not the plain Fisher/expected IRLS weight:
%     h_i     = diag( Wobs^(1/2) X (X'Wobs X)^-1 X' Wobs^(1/2) )
%     pearson_i = (y_i - mu_i) / sqrt(mu_i + alpha*mu_i^2)
%     D_i     = pearson_i^2 * h_i / (k * (1-h_i)^2)
%
% For a log-link NB GLM the observed weight has a closed form
% (derived from statsmodels' GLM.hessian_factor, log-link + NB variance):
%     W_exp_i = mu_i / (1 + alpha*mu_i)                      [expected/Fisher weight]
%     W_obs_i = W_exp_i * (1 + alpha*(y_i-mu_i)/(1+alpha*mu_i))
% The two coincide only when the link is canonical (it isn't, here), so
% using W_exp in place of W_obs silently UNDERSTATES leverage/Cook's D for
% points with large residuals -- this was checked against statsmodels
% directly (session 19 leverage 0.78 / Cook's D 10.75 with W_obs vs. only
% 0.36 / 0.62 if W_exp is used by mistake).
    X = [ones(size(X,1),1), X];
    y = y(:); mu = mu(:);
    k = size(X,2);                          % number of parameters (incl. intercept)

    W_exp = mu ./ (1 + alpha*mu);
    W_obs = W_exp .* (1 + alpha*(y - mu) ./ (1 + alpha*mu));
    W_obs = max(W_obs, 0);                  % numerical safety (weight must be >= 0)

    Wsqrt = sqrt(W_obs);
    Xw = Wsqrt .* X;                        % Wobs^(1/2) * X, row-wise scaling
    H = Xw * ((X' * diag(W_obs) * X) \ Xw');
    leverage = diag(H);
    leverage = min(leverage, 1 - 1e-10);    % numerical safety

    V = mu + alpha * mu.^2;                 % NB variance function
    pearson_resid = (y - mu) ./ sqrt(V);

    cooksD = (pearson_resid.^2 .* leverage) ./ (k * (1 - leverage).^2);
end

function print_table(header, names, beta, se)
    z = beta ./ se;
    pval = 2*(1 - normcdf(abs(z)));
    fprintf('\n--- %s ---\n', header);
    fprintf('%-26s %10s %10s %10s %10s\n','term','coef','se','z','p');
    for i = 1:length(beta)
        fprintf('%-26s %10.3f %10.3f %10.3f %10.3f\n', names{i}, beta(i), se(i), z(i), pval(i));
    end
end

function print_influence(header, sessions, cooksD, leverage)
    fprintf('\n--- %s ---\n', header);
    [sortedD, idx] = sort(cooksD, 'descend');
    fprintf('%-10s %12s %12s\n', 'session', 'cooksD', 'leverage');
    n_show = min(6, length(sessions));
    for i = 1:n_show
        j = idx(i);
        fprintf('%-10d %12.3f %12.3f\n', sessions(j), sortedD(i), leverage(j));
    end
end
