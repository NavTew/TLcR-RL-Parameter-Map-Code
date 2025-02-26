function [im_SR] = TLcR_RL(im_l,YH,YL,upscale,patch_size,overlap,stepsize,window,tau,K,c)
    global zero_matrix
    [imrow, imcol, nTraining] = size(YH);
    Img_SUM      = zeros(imrow,imcol);
    overlap_FLAG = zeros(imrow,imcol);
    
    U = ceil((imrow-overlap)/(patch_size-overlap));  
    V = ceil((imcol-overlap)/(patch_size-overlap)); 
    
    for i = 1:U
         for j = 1:V  
            fprintf('.');
            % obtain the current patch position
            BlockSize  =  GetCurrentBlockSize(imrow,imcol,patch_size,overlap,i,j);    
            if size(YL,1) == size(YH,1)
                BlockSizeS =  GetCurrentBlockSize(imrow,imcol,patch_size,overlap,i,j);  
            else
                BlockSizeS =  GetCurrentBlockSize(size(YL,1),size(YL,2),patch_size/upscale,overlap/upscale,i,j);  
            end
            
            im_l_patch =  im_l(BlockSizeS(1):BlockSizeS(2),BlockSizeS(3):BlockSizeS(4));
            im_l_patch =  im_l_patch(:);   
            im_l_patch = im_l_patch - mean(im_l_patch);
            im_l_patch = [im_l_patch;0;0];
            
            padpixel = (window-patch_size)/stepsize;
            XF = Reshape3D_20Connection(YH,BlockSize,stepsize,padpixel);
            X  = Reshape3D_20Connection_Spatial(YL,BlockSizeS,stepsize,padpixel,c);        
           
            X(1:end-2,:) = X(1:end-2,:) - repmat(mean(X(1:end-2,:)),size(X(1:end-2,:),1),1);
         
            nframe =  size(im_l_patch',1);
            nbase  =  size(X',1);
            XX     =  sum(im_l_patch'.*im_l_patch', 2);        
            SX     =  sum(X'.*X', 2);
            D      =  repmat(XX, 1, nbase) - 2*im_l_patch'*X + repmat(SX', nframe, 1);        
            
            fprintf("DE starting for %d %d\n", i, j);

            % Differential Evolution (DE) Implementation
            pop_size = 10;
            max_gen = 50;
            F = 0.8;
            CR = 0.9;
            K_bounds = [1, 3240];
            
            pop = randi(K_bounds, [pop_size, 1]);
            fitness = arrayfun(@(k) computeError(k, D, X, XF, im_l_patch, tau), pop);
            
            for gen = 1:max_gen
                
                for p = 1:pop_size
                    r = randperm(pop_size, 3);
                    while any(r == p)
                        r = randperm(pop_size, 3);
                    end
                    
                    mutant = pop(r(1)) + F * (pop(r(2)) - pop(r(3)));
                    mutant = max(min(mutant, K_bounds(2)), K_bounds(1));
                    
                    if rand < CR
                        trial = round(mutant);
                    else
                        trial = pop(p);
                    end
                    
                    trial_fitness = computeError(trial, D, X, XF, im_l_patch, tau);
                    
                    if trial_fitness < fitness(p)
                        pop(p) = trial;
                        fitness(p) = trial_fitness;
                    end
                end
            end
            
            [~, best_idx] = min(fitness);
            best_K = pop(best_idx);

            
            old_value = zero_matrix(i, j);
            zero_matrix(i, j) = old_value + best_K;
            
            % Final reconstruction with best_K
            [~, index] = sort(D);        
            Xk  = X(:, index(1:best_K));        
            XFk = XF(:, index(1:best_K));      
            Dk  = D(index(1:best_K));
            
            z   = Xk' - repmat(im_l_patch', best_K, 1);         
            C   = z * z';                                                  
            C   = C + tau * diag(Dk) + eye(best_K, best_K) * (1e-6) * trace(C);   
            w   = C \ ones(best_K,1);  
            w   = w / sum(w);    

            Img  = XFk * w; 
            Img  = reshape(Img, patch_size, patch_size);

            Img_SUM(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4)) = Img_SUM(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4)) + Img;
            overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4)) = overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4)) + 1;
        end
    end
    
    im_SR = Img_SUM ./ overlap_FLAG;
    fprintf('\n');
end

function error = computeError(K, D, X, XF, im_l_patch, tau)
    K = round(K);
    [~, index] = sort(D);        
    Xk  = X(:, index(1:K));        
    XFk = XF(:, index(1:K));      
    Dk  = D(index(1:K));
    
    z   = Xk' - repmat(im_l_patch', K, 1);         
    C   = z * z';                                                  
    C   = C + tau * diag(Dk) + eye(K, K) * (1e-6) * trace(C);   
    w   = C \ ones(K,1);  
    w   = w / sum(w);    
    
    im_l_patch_recon = Xk * w;  
    error = norm(im_l_patch - im_l_patch_recon, 2);
end

