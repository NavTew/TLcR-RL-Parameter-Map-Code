
function [im_SR] = LcRSR(im_l,XH,XL,upscale,patch_size,overlap,tau)

im_l = double(im_l);
[imrow imcol nTraining] = size(XH);

Img_SUM      = zeros(imrow,imcol);
overlap_FLAG = zeros(imrow,imcol);

Lx = ceil((imrow-overlap)/(patch_size-overlap));  %%% the patch 
Ly = ceil((imcol-overlap)/(patch_size-overlap));  %%% �����귶Χ

numm = 0;
% hallucinate the HR patch by patch
for li = 1:Lx
   for lj = 1:Ly     

        BlockSize = GetCurrentBlockSize(imrow,imcol,patch_size,overlap,li,lj);    
        BlockSizeS = GetCurrentBlockSize(imrow/upscale,imcol/upscale,patch_size/upscale,overlap/upscale,li,lj);  
        
        im_l_patch = im_l(BlockSizeS(1):BlockSizeS(2),BlockSizeS(3):BlockSizeS(4));           % extract the patch at position��li,lj��of the input LR face     
        im_l_patch = double(reshape(im_l_patch,patch_size*patch_size/(upscale*upscale),1));   % Reshape 2D image patch into 1D column vectors   
        
        XF = Reshape3D(XH,BlockSize);    % reshape each patch of HR face image to one column
        X  = Reshape3D(XL,BlockSizeS);   % reshape each patch of LR face image to one column

        % represent the LR patch at  position��li,lj��using our LcR 
        nframe=size(im_l_patch',1);
        nbase=size(X',1);
        XX = sum(im_l_patch'.*im_l_patch', 2);        
        SX = sum(X'.*X', 2);
        D  = repmat(XX, 1, nbase)-2*im_l_patch'*X+repmat(SX', nframe, 1); % Calculate the distance between the input LR image patch and the LR training image patches at position��li,lj��
        % Compute the optimal weight vector  for the input LR image patch  with the LR training image patches at position��li,lj��
        z = X' - repmat(im_l_patch', nTraining, 1);         
        C = z*z';                                                
        C = C + tau*diag(D)+eye(nTraining,nTraining)*(1e-6)*trace(C);   
        w = C\ones(nTraining,1);  
        w = w/sum(w); 
  
       
        % obtain the HR patch with the same weight vector w
        Img = XF*w; 
       
        % integrate all the LR patch        
        Img = reshape(Img,patch_size,patch_size);
        Img_SUM(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))      = Img_SUM(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))+Img;
        overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4)) = overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))+1;
    end
end
%  averaging pixel values in the overlapping regions
im_SR = Img_SUM./overlap_FLAG;

