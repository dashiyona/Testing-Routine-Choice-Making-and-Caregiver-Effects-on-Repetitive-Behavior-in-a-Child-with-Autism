%% S003 RRB analysis -- Univariate Screening & Suppression Demonstration
%   Reproduces Supplementary Material Section S5:
%     (1) Each candidate predictor fit ALONE (univariate) against RRB
%         frequency, n=17 (Session 14 excluded), cluster-robust SE.
%     (2) A multivariate model built from ONLY the univariate-significant
%         (p<.05) predictors -- demonstrates that this naive screening
%         approach collapses entirely (nothing survives), because it is
%         blind to Caretaker_Disruption's suppression effect: caretaker
%         looks like a non-predictor alone (p=.64-.71) but is the
%         strongest, cleanest signal once other predictors are held
%         constant (see S003_final_model_and_diagnostics.m).
%
%   Uses the same nbglm_fit / cluster_robust_se conventions as
%   S003_nb_glm_stepwise.m and S003_final_model_and_diagnostics.m.

clear; clc;

%% 1. Load data -----------------------------------------------------------
filename = "path/to/your/file.xlsx";
T = readtable(filename, "Sheet", "Coded Dataset v2");

usable = ismember(T.outcome_status, {'observed','no_RRB_zero'});
T = T(usable, :);

excl14 = T.session ~= 14;
Te = T(excl14, :);                       % n=17, matches main analysis

outcomes = ["obs1", "obs2"];

candidate_names = {'self_routine_disruption','caretaker_disruption','sleep_disruption', ...
                    'feed_disruption','mood_1to5','choice_C'};

%% 2. Univariate models (each predictor alone), n=17 

fprintf(' UNIVARIATE MODELS (each predictor alone), n=17\n');



univariate_results = struct();
for outcome = outcomes
    y = Te.(outcome);
    keep = ~isnan(y);
    yk = y(keep);
    sw = Te.study_week(keep);

    fprintf('\n- Outcome: %s -\n', outcome);
    fprintf('%-28s %10s %10s %10s %10s   %s\n','predictor','coef','se','z','p','flag');
    for i = 1:length(candidate_names)
        pred = candidate_names{i};
        Xk = Te.(pred)(keep);
        [beta, mu, ~] = nbglm_fit(Xk, yk, 1.0);
        se_robust = cluster_robust_se(Xk, yk, mu, 1.0, sw);
        z = beta ./ se_robust;
        pval = 2*(1 - normcdf(abs(z)));
        % row 2 = the predictor's own coefficient (row 1 = intercept)
        flag = "";
        if pval(2) < 0.05, flag = "***"; elseif pval(2) < 0.2, flag = "*"; end
        fprintf('%-28s %10.3f %10.3f %10.3f %10.3f   %s\n', pred, beta(2), se_robust(2), z(2), pval(2), flag);
        univariate_results.(outcome).(pred) = struct('coef', beta(2), 'se', se_robust(2), ...
                                                       'z', z(2), 'p', pval(2));
    end
end

%% 3. Multivariate model from ONLY univariate-significant (p<.05) predictors
% At p<.05 (both observers): self_routine_disruption, feed_disruption, mood_1to5
% Caretaker_Disruption and Choice_C do NOT clear p<.05 alone and are excluded here.
uni_sig_names = {'self_routine_disruption','feed_disruption','mood_1to5'};


fprintf(' MULTIVARIATE MODEL FROM UNIVARIATE-SIGNIFICANT (p<.05) PREDICTORS\n');
fprintf(' Reproduces Supplementary Table S5b -- expect EVERYTHING to lose\n');
fprintf(' significance (this demonstrates the suppression effect, it is\n');
fprintf(' not the recommended model -- see the FINAL model script instead)\n');


names_screened = [{'Intercept'}, uni_sig_names];
for outcome = outcomes
    y = Te.(outcome);
    keep = ~isnan(y);
    yk = y(keep);
    Xk = [Te.self_routine_disruption(keep), Te.feed_disruption(keep), Te.mood_1to5(keep)];

    [beta, mu, ~] = nbglm_fit(Xk, yk, 1.0);
    se_robust = cluster_robust_se(Xk, yk, mu, 1.0, Te.study_week(keep));
    print_table(outcome + " -- univariate-screened model (n=" + sum(keep) + ")", names_screened, beta, se_robust);
end



Xf = [Te.self_routine_disruption(keep), Te.feed_disruption(keep), Te.choice_C(keep)];
R = corrcoef(Xf);
vif = diag(inv(R))

%% ========================================================================
%  LOCAL FUNCTIONS (identical to S003_final_model_and_diagnostics.m)
%  ========================================================================

function [beta, mu, W] = nbglm_fit(X, y, alpha)
% Fixed-dispersion Negative Binomial GLM, log link, fit via IRLS.
% Variance function: V(mu) = mu + alpha*mu^2   (alpha fixed, not estimated)
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

function print_table(header, names, beta, se)
    z = beta ./ se;
    pval = 2*(1 - normcdf(abs(z)));
    fprintf('\n--- %s ---\n', header);
    fprintf('%-26s %10s %10s %10s %10s\n','term','coef','se','z','p');
    for i = 1:length(beta)
        fprintf('%-26s %10.3f %10.3f %10.3f %10.3f\n', names{i}, beta(i), se(i), z(i), pval(i));
    end
end
