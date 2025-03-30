function [im_SR] = Greywolf(im_l, YH, YL, upscale, patch_size, overlap, stepsize, window, tau, K, c)

    global zero_matrix
    [imrow, imcol, nTraining] = size(YH);
    Img_SUM      = zeros(imrow, imcol);
    overlap_FLAG = zeros(imrow, imcol);
    
    U = ceil((imrow - overlap) / (patch_size - overlap));  
    V = ceil((imcol - overlap) / (patch_size - overlap)); 
    
    for i = 1:U
        fprintf('.');
        for j = 1:V  
            
            BlockSize  = GetCurrentBlockSize(imrow, imcol, patch_size, overlap, i, j);    
            if size(YL,1) == size(YH,1)
                BlockSizeS = GetCurrentBlockSize(imrow, imcol, patch_size, overlap, i, j);  
            else
                BlockSizeS = GetCurrentBlockSize(size(YL,1), size(YL,2), patch_size/upscale, overlap/upscale, i, j);  
            end
            
            im_l_patch = im_l(BlockSizeS(1):BlockSizeS(2), BlockSizeS(3):BlockSizeS(4));
            im_l_patch = im_l_patch(:);   
            im_l_patch = im_l_patch - mean(im_l_patch);
            im_l_patch = [im_l_patch; 0; 0]; 
            
            padpixel = (window - patch_size) / stepsize;
            XF = Reshape3D_20Connection(YH, BlockSize, stepsize, padpixel);
            X  = Reshape3D_20Connection_Spatial(YL, BlockSizeS, stepsize, padpixel, c);        
           
            X(1:end-2, :) = X(1:end-2, :) - repmat(mean(X(1:end-2, :)), size(X(1:end-2, :), 1), 1);
         
            nframe = size(im_l_patch', 1);
            nbase  = size(X', 1);
            XX     = sum(im_l_patch'.*im_l_patch', 2);        
            SX     = sum(X'.*X', 2);
            D      = repmat(XX, 1, nbase) - 2*im_l_patch' * X + repmat(SX', nframe, 1);

            identical_patch_idx = find(abs(D) < 1e-8);
            valid_idx = setdiff(1:size(D, 2), identical_patch_idx);

            if isempty(valid_idx)
                error('All training patches are identical to input. Adjust training set.');
            end

            D_filtered = D(valid_idx);
            X_filtered = X(:, valid_idx);
            XF_filtered = XF(:, valid_idx);

            % Grey Wolf Optimizer (GWO) Implementation
            wolf_num = 10;
            max_iter = 40;
            K_bounds = [1, 197];
            
            alpha = inf;
            beta = inf;
            delta = inf;
            
            wolves = randi(K_bounds, [wolf_num, 1]);
            fitness = arrayfun(@(k) computeError(k, D_filtered, X_filtered, XF_filtered, im_l_patch, tau), wolves);
            
            for iter = 1:max_iter
                [sorted_fitness, idx] = sort(fitness);
                alpha = wolves(idx(1));
                beta = wolves(idx(2));
                delta = wolves(idx(3));
                
                for w = 1:wolf_num
                    A1 = 2 * rand - 1;
                    C1 = 2 * rand;
                    D_alpha = abs(C1 * alpha - wolves(w));
                    X1 = alpha - A1 * D_alpha;
                    
                    A2 = 2 * rand - 1;
                    C2 = 2 * rand;
                    D_beta = abs(C2 * beta - wolves(w));
                    X2 = beta - A2 * D_beta;
                    
                    A3 = 2 * rand - 1;
                    C3 = 2 * rand;
                    D_delta = abs(C3 * delta - wolves(w));
                    X3 = delta - A3 * D_delta;
                    
                    new_wolf = (X1 + X2 + X3) / 3;
                    new_wolf = round(max(min(new_wolf, K_bounds(2)), K_bounds(1)));
                    
                    new_fitness = computeError(new_wolf, D_filtered, X_filtered, XF_filtered, im_l_patch, tau);
                    
                    if new_fitness < fitness(w)
                        wolves(w) = new_wolf;
                        fitness(w) = new_fitness;
                    end
                end
            end
            
            best_K = wolves(find(fitness == min(fitness), 1));
            fprintf('%d\n', best_K);
            zero_matrix(i,j) = zero_matrix(i,j) + best_K;
            
            [~, index] = sort(D_filtered);
            Xk  = X_filtered(:, index(1:best_K));
            XFk = XF_filtered(:, index(1:best_K));
            Dk  = D_filtered(index(1:best_K));
            
            z   = Xk' - repmat(im_l_patch', best_K, 1);         
            C   = z * z';                                                  
            C   = C + tau * diag(Dk) + eye(best_K, best_K) * (1e-6) * trace(C);   
            w   = C \ ones(best_K, 1);  
            w   = w / sum(w);    
            
            Img  = XFk * w; 
            Img  = reshape(Img, patch_size, patch_size);

            Img_SUM(BlockSize(1):BlockSize(2), BlockSize(3):BlockSize(4)) = Img_SUM(BlockSize(1):BlockSize(2), BlockSize(3):BlockSize(4)) + Img;
            overlap_FLAG(BlockSize(1):BlockSize(2), BlockSize(3):BlockSize(4)) = overlap_FLAG(BlockSize(1):BlockSize(2), BlockSize(3):BlockSize(4)) + 1;
        end
    end
    
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