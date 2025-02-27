function [im_SR] = TLcR_RL(im_l, YH, YL, upscale, patch_size, overlap, stepsize, window, tau, K, c)

    global zero_matrix
    [imrow, imcol, nTraining] = size(YH);
    Img_SUM      = zeros(imrow, imcol);
    overlap_FLAG = zeros(imrow, imcol);
    
    U = ceil((imrow - overlap) / (patch_size - overlap));  
    V = ceil((imcol - overlap) / (patch_size - overlap)); 
    
    for i = 1:U
        fprintf('.');
        for j = 1:V  
            
            
            % Obtain the current patch position
            BlockSize  = GetCurrentBlockSize(imrow, imcol, patch_size, overlap, i, j);    
            if size(YL,1) == size(YH,1)
                BlockSizeS = GetCurrentBlockSize(imrow, imcol, patch_size, overlap, i, j);  
            else
                BlockSizeS = GetCurrentBlockSize(size(YL,1), size(YL,2), patch_size/upscale, overlap/upscale, i, j);  
            end
            
            % Extract the low-resolution patch
            im_l_patch = im_l(BlockSizeS(1):BlockSizeS(2), BlockSizeS(3):BlockSizeS(4));
            im_l_patch = im_l_patch(:);   
            im_l_patch = im_l_patch - mean(im_l_patch); % Normalize by subtracting mean
            im_l_patch = [im_l_patch; 0; 0]; % Add spatial information
            
            % Reshape training data
            padpixel = (window - patch_size) / stepsize;
            XF = Reshape3D_20Connection(YH, BlockSize, stepsize, padpixel);
            X  = Reshape3D_20Connection_Spatial(YL, BlockSizeS, stepsize, padpixel, c);        
           
            % Normalize training patches
            X(1:end-2, :) = X(1:end-2, :) - repmat(mean(X(1:end-2, :)), size(X(1:end-2, :), 1), 1);
         
            % Compute distance between input patch and training patches
            nframe = size(im_l_patch', 1);
            nbase  = size(X', 1);
            XX     = sum(im_l_patch'.*im_l_patch', 2);        
            SX     = sum(X'.*X', 2);
            D      = repmat(XX, 1, nbase) - 2*im_l_patch' * X + repmat(SX', nframe, 1);        

            % Identify and remove identical patches
            identical_patch_idx = find(abs(D) < 1e-8); % Avoid floating-point precision errors
            valid_idx = setdiff(1:size(D, 2), identical_patch_idx); % Indices of non-identical patches

            % Ensure there are still valid patches left
            if isempty(valid_idx)
                error('All training patches are identical to input. Adjust training set.');
            end

            % Use only non-identical patches
            D_filtered = D(valid_idx);
            X_filtered = X(:, valid_idx);
            XF_filtered = XF(:, valid_idx);

            % Differential Evolution (DE) Implementation
            pop_size = 10; % Population size
            max_gen = 50; % Maximum generations
            F = 0.8; % Mutation factor
            CR = 0.9; % Crossover rate
            K_bounds = [1, 44]; % Range of K values
            
            % Initialize population with random values in range
            pop = randi(K_bounds, [pop_size, 1]);
            fitness = arrayfun(@(k) computeError(k, D_filtered, X_filtered, XF_filtered, im_l_patch, tau), pop);
            
            for gen = 1:max_gen
                for p = 1:pop_size
                    % Select three random individuals (different from p)
                    r = randperm(pop_size, 3);
                    while any(r == p)
                        r = randperm(pop_size, 3);
                    end
                    
                    % Perform mutation and crossover
                    mutant = pop(r(1)) + F * (pop(r(2)) - pop(r(3))); % Mutation
                    mutant = max(min(mutant, K_bounds(2)), K_bounds(1)); % Bound mutation
                    
                    if rand < CR
                        trial = round(mutant);
                    else
                        trial = pop(p);
                    end
                    
                    % Evaluate fitness
                    trial_fitness = computeError(trial, D_filtered, X_filtered, XF_filtered, im_l_patch, tau);
                    
                    % Selection: Replace if new candidate is better
                    if trial_fitness < fitness(p)
                        pop(p) = trial;
                        fitness(p) = trial_fitness;
                    end
                end
            end
            
            % Select the best K from the evolved population
            [~, best_idx] = min(fitness);
            best_K = pop(best_idx);
            fprintf('%d\n', best_K);
            

            zero_matrix(i,j) = zero_matrix(i,j) + best_K;
            % Perform final reconstruction using best_K
            [~, index] = sort(D_filtered);
            Xk  = X_filtered(:, index(1:best_K));
            XFk = XF_filtered(:, index(1:best_K));
            Dk  = D_filtered(index(1:best_K));
            
            % Compute weights
            z   = Xk' - repmat(im_l_patch', best_K, 1);         
            C   = z * z';                                                  
            C   = C + tau * diag(Dk) + eye(best_K, best_K) * (1e-6) * trace(C);   
            w   = C \ ones(best_K, 1);  
            w   = w / sum(w);    

            % Compute high-resolution patch
            Img  = XFk * w; 
            Img  = reshape(Img, patch_size, patch_size);

            % Aggregate reconstructed patches
            Img_SUM(BlockSize(1):BlockSize(2), BlockSize(3):BlockSize(4)) = Img_SUM(BlockSize(1):BlockSize(2), BlockSize(3):BlockSize(4)) + Img;
            overlap_FLAG(BlockSize(1):BlockSize(2), BlockSize(3):BlockSize(4)) = overlap_FLAG(BlockSize(1):BlockSize(2), BlockSize(3):BlockSize(4)) + 1;
        end
    end
    
    % Compute final super-resolved image by averaging overlapping regions
    im_SR = Img_SUM ./ overlap_FLAG;
    fprintf('\n');
end

% Function to compute reconstruction error
function error = computeError(K, D, X, XF, im_l_patch, tau)
    K = round(K);
    [~, index] = sort(D);        
    Xk  = X(:, index(1:K));        
    XFk = XF(:, index(1:K));      
    Dk  = D(index(1:K));
    
    % Compute weight vector
    z   = Xk' - repmat(im_l_patch', K, 1);         
    C   = z * z';                                                  
    C   = C + tau * diag(Dk) + eye(K, K) * (1e-6) * trace(C);   
    w   = C \ ones(K, 1);  
    w   = w / sum(w);    
    
    % Compute reconstruction error
    im_l_patch_recon = Xk * w;  
    error = norm(im_l_patch - im_l_patch_recon, 2);
end
