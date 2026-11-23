  %% S003 RRB analysis -- SERIAL-DEPENDENCE-ADJUSTED MODEL + EXACT BLOCK PERMUTATION (v2)
%
% 
%    1. Exact block permutation test: all G! orderings of the study-week blocks are
%       enumerated (8! = 40,320 here), so p values carry
%       no Monte Carlo noise. The identity ordering is included, so p >= 1/G!.
%       (If G > 9, the script falls back to N_RANDOM random orderings and uses
%       p = (count + 1) / (N + 1).)
%    2. Stage-1 convergence check: the IRLS fit now reports whether it converged and
%       the script warns if it did not.
%    3. Leave-one-session-out sensitivity now also re-runs the exact block test, so
%       the robustness of each p value (not only of each coefficient) is reported.
%    4. Results, jackknife and influence tables are written to CSV.
%    5. Cook's distance and Leverage are reported.
% 
%    The Model & Testing::
%    The model and test: stage 1 removes the time trend and the
%    lag-1/lag-2 counts from the outcome; stage 2 regresses the Pearson residuals on
%    the four predictors; predictor rows are reshuffled as intact study-week blocks.
%    Adding session to stage 2 with predictors reshuffled, or Freedman-Lane residual reshuffling, was
%    MORE liberal (9-18%). Read p values near .05 with that in mind; the simulation-
%    calibrated 5% threshold for this design is p < .025-.037.
%
%  Note on unequal blocks: week 1 has three sessions and the others two. Stacking
%  reshuffled blocks therefore shifts row boundaries for some orderings. 

clear; clc;

%% 0. Settings ------------------------------------------------------------
filename      = "path/to/your/dataa.xlsx";
sheetname     = 'Coded Dataset v2';
outcome_names = {'obs1', 'obs2'};
pred_names    = {'caretaker_disruption','feed_disruption','mood_1to5','choice_C'};
exclude_stage2 = 14;          % session excluded from stage 2 (kept in stage 1, as in v1)
alpha_val     = 1.0;          % fixed NB dispersion, as in v1
MAX_EXACT_G   = 9;            % enumerate all orderings when G <= this
N_RANDOM      = 10000;        % otherwise sample this many orderings
MAKE_FIGS     = true;
out_prefix    = 'S003_v2_';
rng(42);

%% 1. Load data and convert to plain arrays 
% LOAD-BEGIN

T = readtable(filename, 'Sheet', sheetname);
usable = ismember(T.outcome_status, {'observed','no_RRB_zero'});
T = sortrows(T(usable, :), 'session');
session = T.session;
week    = T.study_week;
Y       = zeros(height(T), numel(outcome_names));
for k = 1:numel(outcome_names), Y(:,k) = T.(outcome_names{k}); end
Xall    = zeros(height(T), numel(pred_names));
for k = 1:numel(pred_names), Xall(:,k) = T.(pred_names{k}); end
% LOAD-END

names_final = [{'Intercept'}, pred_names];
n_total = numel(session);

%% 2. Stage 1: remove time trend and lag-1/lag-2 dependence (Pearson residuals) --

fprintf(' STAGE 1: AR(2) + TREND PRE-WHITENED RESIDUALS\n');

R = nan(n_total, numel(outcome_names));
for k = 1:numel(outcome_names)
    y_raw  = Y(:,k);
    y_lag1 = [mean(y_raw, 'omitnan'); y_raw(1:end-1)];
    y_lag2 = [mean(y_raw, 'omitnan'); mean(y_raw, 'omitnan'); y_raw(1:end-2)];
    X_base = [session, y_lag1, y_lag2];
    keep_b = ~isnan(y_raw);
    [beta_base, ~, converged] = nbglm_fit(X_base(keep_b,:), y_raw(keep_b), alpha_val);
    if ~converged
        warning('Stage-1 NB fit did not converge for %s; residuals may be unreliable.', outcome_names{k});
    end
    mu = max(exp([ones(n_total,1), X_base] * beta_base), 1e-8);
    R(:,k) = (y_raw - mu) ./ sqrt(mu + alpha_val * mu.^2);
    fprintf('%-14s converged = %d\n', outcome_names{k}, converged);
end

%% 3. Stage 2: final model with exact block permutation ---------------------
in2   = session ~= exclude_stage2;
X2    = Xall(in2,:);  s2 = session(in2);  w2 = week(in2);  R2 = R(in2,:);

fprintf(' STAGE 2: SERIAL-DEPENDENCE-ADJUSTED MODEL, EXACT BLOCK PERMUTATION (n=%d)\n', sum(in2));


results = struct();
for k = 1:numel(outcome_names)
    oc = outcome_names{k};
    keep = ~isnan(R2(:,k));
    Xk = X2(keep,:);  yk = R2(keep,k);  sess_k = s2(keep);  groups = w2(keep);
    Xk_int = [ones(sum(keep),1), Xk];

    beta_obs = Xk_int \ yk;
    se_rob   = cluster_se(Xk_int, yk, beta_obs, groups);
    [p_block, null_beta, n_ord, exact] = block_perm_test(Xk_int, yk, groups, beta_obs, MAX_EXACT_G, N_RANDOM);
    null_lo = arrayfun(@(i) percentile(null_beta(:,i), 2.5),  1:numel(beta_obs))';
    null_hi = arrayfun(@(i) percentile(null_beta(:,i), 97.5), 1:numel(beta_obs))';

    print_table([oc ' -- serial-dependence-adjusted model'], names_final, beta_obs, se_rob, p_block, null_lo, null_hi);
    if exact
        fprintf('  (exact test: all %d orderings of %d week blocks)\n', n_ord, numel(unique(groups)));
    else
        fprintf('  (random test: %d orderings, p = (count+1)/(N+1))\n', n_ord);
    end

    % -- Leave-one-session-out: coefficients AND exact block p --
    fprintf('Leave-one-session-out sensitivity (coefficient and block p for each dropped session)...\n');
    us = unique(sess_k);  J = numel(us);
    jk_beta = zeros(J, numel(beta_obs));  jk_p = zeros(J, numel(beta_obs));
    for j = 1:J
        ks = sess_k ~= us(j);
        bj = Xk_int(ks,:) \ yk(ks);
        jk_beta(j,:) = bj';
        jk_p(j,:) = block_perm_test(Xk_int(ks,:), yk(ks), groups(ks), bj, MAX_EXACT_G, N_RANDOM)';
    end
    for i = 2:numel(beta_obs)
        [~, jmin] = min(jk_beta(:,i)); [~, jmax] = max(jk_beta(:,i)); [pmax, jp] = max(jk_p(:,i));
        fprintf('  %-22s coef range %.3f (drop %d) to %.3f (drop %d); sign changes: %d; largest p %.3f (drop %d)\n', ...
            names_final{i}, jk_beta(jmin,i), us(jmin), jk_beta(jmax,i), us(jmax), ...
            sum(sign(jk_beta(:,i)) ~= sign(beta_obs(i))), pmax, us(jp));
    end

    % --- Influence (Cook's distance, leverage) ---
    kk = size(Xk_int,2);
    H = Xk_int * ((Xk_int' * Xk_int) \ Xk_int');
    lev = min(diag(H), 1 - 1e-10);
    res = yk - Xk_int * beta_obs;
    s2hat = sum(res.^2) / (numel(yk) - kk);
    cooksD = (res.^2 ./ (kk * s2hat)) .* (lev ./ (1 - lev).^2);
    print_influence([oc ' -- influence'], sess_k, cooksD, lev, kk);

    % --- Save ---
    write_csv([out_prefix oc '_results.csv'], ...
        {'term','coef','se_cluster','block_p','null_2.5','null_97.5'}, names_final', [beta_obs, se_rob, p_block, null_lo, null_hi]);
    write_csv([out_prefix oc '_jackknife.csv'], ...
        [{'dropped_session'}, strcat(names_final, '_coef'), strcat(names_final, '_block_p')], ...
        cellstr(num2str(us)), [jk_beta, jk_p]);
    write_csv([out_prefix oc '_influence.csv'], {'session','cooksD','leverage'}, ...
        cellstr(num2str(sess_k)), [cooksD, lev]);
    results.(oc) = struct('beta', beta_obs, 'se', se_rob, 'p', p_block, ...
        'jk_beta', jk_beta, 'jk_p', jk_p, 'cooksD', cooksD, 'leverage', lev, 'sessions', sess_k);

    % -- Figures --
    if MAKE_FIGS
        figure('Name', ['Block null -- ' oc], 'Position', [100 100 820 620]);
        for i = 2:numel(beta_obs)
            subplot(2,2,i-1);
            histogram(null_beta(:,i), 40, 'FaceColor', [0.4 0.5 0.65], 'EdgeColor', 'none'); hold on;
            yl = ylim;
            plot([null_lo(i) null_lo(i)], yl, 'b--', 'LineWidth', 1.1);
            plot([null_hi(i) null_hi(i)], yl, 'b--', 'LineWidth', 1.1);
            plot([beta_obs(i) beta_obs(i)], yl, 'r-', 'LineWidth', 2.5);
            title(names_final{i}, 'Interpreter', 'none');
            xlabel('Coefficient under week-block reshuffling'); ylabel('Count'); grid on;
            legend({'Null (block reshuffles)', 'Null 2.5/97.5%', '', ...
                sprintf('Observed %.3f, p = %.3f', beta_obs(i), p_block(i))}, 'Location', 'best', 'FontSize', 8);
        end
        sgtitle(['Observed coefficients against the week-block null: ' oc], 'Interpreter', 'none');

        figure('Name', ['Jackknife -- ' oc], 'Position', [150 150 820 620]);
        for i = 2:numel(beta_obs)
            subplot(2,2,i-1);
            yyaxis left;  plot(1:J, jk_beta(:,i), 'o-', 'LineWidth', 1.4); hold on;
            plot([1 J], [beta_obs(i) beta_obs(i)], 'r--', 'LineWidth', 1.2); ylabel('Coefficient');
            yyaxis right; bar(1:J, jk_p(:,i), 0.4, 'FaceAlpha', 0.25, 'EdgeColor', 'none'); ylim([0 1]); ylabel('Block p');
            xticks(1:J); xticklabels(cellstr(num2str(us))); xlabel('Dropped session');
            title(names_final{i}, 'Interpreter', 'none'); grid on;
        end
        sgtitle(['Leave-one-session-out: coefficient (line) and block p (bars): ' oc], 'Interpreter', 'none');
    end
end
fprintf('\nDone. Tables written with prefix "%s".\n', out_prefix);


%%  LOCAL FUNCTIONS

function [beta, mu, converged] = nbglm_fit(X, y, alpha)
% Fixed-dispersion negative binomial GLM, log link, IRLS.
X = [ones(size(X,1),1), X]; y = y(:); n = numel(y);
beta = zeros(size(X,2),1); beta(1) = log(max(mean(y), 0.1));
mu = max(mean(y), 0.1) * ones(n,1); eta = log(mu); converged = false;
for iter = 1:100
    W = mu ./ (1 + alpha*mu);
    z = eta + (y - mu) ./ mu;
    beta_new = (X' * (W .* X)) \ (X' * (W .* z));
    eta = X * beta_new; mu = max(exp(eta), 1e-8);
    if max(abs(beta_new - beta)) < 1e-8
        beta = beta_new; converged = true; break
    end
    beta = beta_new;
end
end

function se = cluster_se(X, y, beta, groups)
% Cluster-robust (sandwich) SE by study week. Descriptive only: with ~8 clusters
% these are too small, which is why inference uses the block permutation test.
res = y - X*beta; ug = unique(groups); G = numel(ug); [n, k] = size(X);
bread = inv(X' * X); meat = zeros(k);
for g = 1:G
    idx = groups == ug(g); sg = X(idx,:)' * res(idx); meat = meat + sg * sg';
end
c = (G/(G-1)) * ((n-1)/(n-k));
se = sqrt(diag(c * (bread * meat * bread)));
end

function [p, null_beta, n_ord, exact] = block_perm_test(X, y, groups, beta_obs, max_exact_G, n_random)
% Reshuffle predictor rows as intact blocks (study weeks) relative to y.
% Column 1 (intercept) stays fixed. Exact when G <= max_exact_G.
ug = unique(groups); G = numel(ug);
blocks = cell(G,1); for g = 1:G, blocks{g} = find(groups == ug(g)); end
exact = G <= max_exact_G;
if exact
    orders = perms(1:G); n_ord = size(orders,1);
else
    n_ord = n_random; orders = zeros(n_ord, G);
    for r = 1:n_ord, orders(r,:) = randperm(G); end
end
null_beta = zeros(n_ord, numel(beta_obs));
for r = 1:n_ord
    rows = vertcat(blocks{orders(r,:)});
    Xp = X(rows,:); Xp(:,1) = 1;
    null_beta(r,:) = (Xp \ y)';
end
tol = 1e-12;
cnt = sum(abs(null_beta) >= abs(beta_obs') - tol, 1)';
if exact
    p = cnt / n_ord;                 % identity ordering is among the enumerated ones
else
    p = (cnt + 1) / (n_ord + 1);
end
end

function v = percentile(x, q)
% Linear-interpolation percentile without toolbox dependency.
x = sort(x(:)); n = numel(x); idx = (q/100)*(n-1) + 1;
lo = floor(idx); hi = ceil(idx);
v = x(lo) + (idx - lo) * (x(hi) - x(lo));
end

function print_table(header, names, beta, se, p, lo, hi)
fprintf('\n--- %s ---\n', header);
fprintf('%-24s %9s %10s %9s %10s %20s\n', 'term', 'coef', 'se(clust)', 'z', 'block_p', 'null 95% interval');
for i = 1:numel(beta)
    fprintf('%-24s %9.3f %10.3f %9.3f %10.3f   [%7.3f, %7.3f]\n', names{i}, beta(i), se(i), beta(i)/se(i), p(i), lo(i), hi(i));
end
end

function print_influence(header, sessions, cooksD, lev, k)
n = numel(sessions);
fprintf('\n--- %s (Cook''s D threshold 4/n = %.3f; leverage threshold 2k/n = %.3f) ---\n', header, 4/n, 2*k/n);
[d, idx] = sort(cooksD, 'descend');
fprintf('%-10s %10s %10s\n', 'session', 'cooksD', 'leverage');
for i = 1:min(6, n)
    fprintf('%-10d %10.3f %10.3f\n', sessions(idx(i)), d(i), lev(idx(i)));
end
end

function write_csv(fname, header, rowlabels, M)
fid = fopen(fname, 'w');
fprintf(fid, '%s', header{1}); fprintf(fid, ',%s', header{2:end}); fprintf(fid, '\n');
for r = 1:size(M,1)
    fprintf(fid, '%s', strtrim(rowlabels{r})); fprintf(fid, ',%.6g', M(r,:)); fprintf(fid, '\n');
end
fclose(fid);
end
