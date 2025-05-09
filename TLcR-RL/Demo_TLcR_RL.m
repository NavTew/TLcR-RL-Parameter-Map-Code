
% =========================================================================
% Simple demo codes for face image super-resolution via TLcR-RL
%=========================================================================

clc;close all;clear all;
addpath('.\utilities');


nTraining   = 22;        % number of training sample
nTesting    = 10;         % number of ptest sample
upscale     = 4;          % upscaling factor 
patch_size  = 12;         % image patch size
overlap     = 4;          % the overlap between neighborhood patches
stepsize    = 2;          % step size

% parameter settings
window      = 20;         % contextal patch,12, 16, 20, 24, 28, ... (12 means us no contextal information)
K           = 3;        % thresholding parameter
tau         = 0.04;       % locality constraint parameter
layer       = 5;          % the iteration value in reproducing learning
c           = 10;         % the weight of the spatial feature

% construct the HR and LR training pairs from the FEI face database
[YH YL] = Training_LH(upscale,nTraining);

[imrow, imcol, nTraining] = size(YH);
Img_SUM      = zeros(imrow, imcol);
overlap_FLAG = zeros(imrow, imcol);

U = ceil((imrow - overlap) / (patch_size - overlap));  
V = ceil((imcol - overlap) / (patch_size - overlap));  

disp(U);
disp(V);

global zero_matrix;
zero_matrix = zeros(U, V);




fprintf('\nStarting to make K-Map')


for TrainImgIndex = 1:nTraining
    fprintf('\nProcessing  %d_train.jpg\n', TrainImgIndex);

    % read ground truth of one test face 
    strh    = strcat('.\trainingFaces\',num2str(TrainImgIndex),'_h.jpg');
    im_h    = double(imread(strh));

    % generate the input LR face by smooth and down-sampleing
    psf         = fspecial('average', [4 4]); 
    im_s    = imfilter(im_h,psf);
    im_l    = imresize(im_s,1/upscale,'bicubic');
    % upscale the LR face to HR size
    im_b = imresize(im_l,upscale,'bicubic');
    
    tic;
    % hallucinate the high frequency face via TLcR



    %[im_SR] = TLcR_RL(im_b,YH,YL,upscale,patch_size,overlap,stepsize,window,tau,K,c); 
    [im_SR] = Greywolf(im_b,YH,YL,upscale,patch_size,overlap,stepsize,window,tau,K,c); 
    
    

    % add the high frequency face to result
    %[im_SR] = im_SR+im_b;
    %cputime(TrainImgIndex) = toc;
    % only for test images do we need imwrite
    % imwrite(uint8(im_SR),strcat('./results/',num2str(TrainImgIndex),'_',num2str(tau),'_TLcR.bmp'),'bmp');  

    % compute PSNR and SSIM for Bicubi,c and TLcR method
    % bicubic_psnr(TrainImgIndex) = psnr(im_b,im_h);
    % bicubic_ssim(TrainImgIndex) = ssim(im_b,im_h);

    % TLcR_psnr(TrainImgIndex) = psnr(im_SR,im_h);
    % TLcR_ssim(TrainImgIndex) = ssim(im_SR,im_h);  


    % % updata the result by reproducing learning
    % for ls = 1:layer
    %     im_lSR  = imfilter(im_SR,psf);
    %     im_lSR  = imresize(im_lSR,1/upscale,'bicubic');    
    %     im_lSR  = imresize(im_lSR,size(im_SR));
    %     [im_SR] = TLcR_RL(im_b,cat(3,YH,im_SR-im_lSR),cat(3,YL,im_lSR),upscale,patch_size,overlap,stepsize,window,tau,K,c);
    %     [im_SR] = im_SR+im_b;
    %     % compute PSNR and SSIM for Bicubic and TLcR-RL method
    %     TLcRRL_psnr(ls,TrainImgIndex) = psnr(im_SR,im_h);
    %     TLcRRL_ssim(ls,TrainImgIndex) = ssim(im_SR,im_h);          
    % end 
    % imwrite(uint8(im_SR),strcat('./results/',num2str(TrainImgIndex),'_',num2str(tau),'_TLcRRL.bmp'),'bmp');  
    
    % display the objective results (PSNR and SSIM)
    % fprintf('PSNR for Bicubic:  %f dB\n', bicubic_psnr(TrainImgIndex));
    % fprintf('PSNR for TLcR:     %f dB\n', TLcR_psnr(TrainImgIndex));
%    fprintf('PSNR for TLcR-RL:  %f dB\n', TLcRRL_psnr(layer,TrainImgIndex));
    % fprintf('SSIM for Bicubic:  %f dB\n', bicubic_ssim(TrainImgIndex));
    % fprintf('SSIM for TLcR:     %f dB\n', TLcR_ssim(TrainImgIndex));
%    fprintf('SSIM for TLcR-RL:  %f dB\n', TLcRRL_ssim(layer,TrainImgIndex));
    % disp(zero_matrix);
end 
disp(zero_matrix)
zero_matrix = round(zero_matrix / nTraining);
%put 360
disp(zero_matrix)

fprintf("\nK map made, now testing begins\n")

% nTesting = 11;
for TestImgIndex = 1:nTesting
    fprintf('\nProcessing  %d_test.jpg\n', TestImgIndex); 
    strh    = strcat('.\testFaces\',num2str(TestImgIndex),'_test.jpg');
    im_h    = double(imread(strh));
    psf         = fspecial('average', [4 4]); 
    im_s    = imfilter(im_h,psf);
    im_l    = imresize(im_s,1/upscale,'bicubic');
    im_b = imresize(im_l,upscale,'bicubic');
    
    tic;
    % hallucinate the high frequency face via TLcR
    [im_SR] = my_TLcR_RL(im_b,YH,YL,upscale,patch_size,overlap,stepsize,window,tau,K,c);     
    % add the high frequency face to result
    [im_SR] = im_SR+im_b;
    cputime(TestImgIndex) = toc;
    imwrite(uint8(im_SR),strcat('./results/',num2str(TestImgIndex),'_',num2str(tau),'_TLcR.bmp'),'bmp');  

    % compute PSNR and SSIM for Bicubic and TLcR method
    bicubic_psnr(TestImgIndex) = psnr(im_b,im_h);
    bicubic_ssim(TestImgIndex) = ssim(im_b,im_h);
    TLcR_psnr(TestImgIndex) = psnr(im_SR,im_h);
    TLcR_ssim(TestImgIndex) = ssim(im_SR,im_h);  


    % % updata the result by reproducing learning
    % for ls = 1:layer
    %     im_lSR  = imfilter(im_SR,psf);
    %     im_lSR  = imresize(im_lSR,1/upscale,'bicubic');    
    %     im_lSR  = imresize(im_lSR,size(im_SR));
    %     [im_SR] = TLcR_RL(im_b,cat(3,YH,im_SR-im_lSR),cat(3,YL,im_lSR),upscale,patch_size,overlap,stepsize,window,tau,K,c);
    %     [im_SR] = im_SR+im_b;
    %     % compute PSNR and SSIM for Bicubic and TLcR-RL method
    %     TLcRRL_psnr(ls,TrainImgIndex) = psnr(im_SR,im_h);
    %     TLcRRL_ssim(ls,TrainImgIndex) = ssim(im_SR,im_h);          
    % end 
    %imwrite(uint8(im_SR),strcat('./results/',num2str(TrainImgIndex),'_',num2str(tau),'_TLcRRL.bmp'),'bmp');  

    
    % display the objective results (PSNR and SSIM)
    fprintf('PSNR for Bicubic:  %f dB\n', bicubic_psnr(TestImgIndex));
    fprintf('PSNR for TLcR:     %f dB\n', TLcR_psnr(TestImgIndex));
   %fprintf('PSNR for TLcR-RL:  %f dB\n', TLcRRL_psnr(layer,TestImgIndex));
    fprintf('SSIM for Bicubic:  %f dB\n', bicubic_ssim(TestImgIndex));
    fprintf('SSIM for TLcR:     %f dB\n', TLcR_ssim(TestImgIndex));
   %fprintf('SSIM for TLcR-RL:  %f dB\n', TLcRRL_ssim(layer,TestImgIndex));
    

end

disp(zero_matrix);

fprintf('===============================================\n');
fprintf('Average PSNR for Bicubic:  %f dB\n', sum(bicubic_psnr)/nTesting);
fprintf('Average PSNR for TLcR:     %f dB\n', sum(TLcR_psnr)/nTesting);
%fprintf('Average PSNR for TLcRRL:   %f dB\n', sum(TLcRRL_psnr(layer,:))/nTesting);
fprintf('Average SSIM for Bicubic:  %f dB\n', sum(bicubic_ssim)/nTesting);
fprintf('Average SSIM for TLcR:     %f dB\n', sum(TLcR_ssim)/nTesting);
%fprintf('Average SSIM for TLcR-RL:     %f dB\n', sum(TLcRRL_ssim(layer,:))/nTesting);
fprintf('===============================================\n');




































% % =========================================================================
% % Simple demo codes for face image super-resolution via TLcR-RL
% %=========================================================================

% clc;close all;clear all;
% addpath('.\utilities');

% nTraining   = 22;        % number of training sample
% nTesting    = 10;         % number of ptest sample
% upscale     = 4;          % upscaling factor 
% patch_size  = 12;         % image patch size
% overlap     = 4;          % the overlap between neighborhood patches
% stepsize    = 2;          % step size

% % parameter settings
% window      = 20;         % contextal patch,12, 16, 20, 24, 28, ... (12 means us no contextal information)
% K           = 32;        % thresholding parameter
% tau         = 0.04;       % locality constraint parameter
% layer       = 5;          % the iteration value in reproducing learning
% c           = 10;         % the weight of the spatial feature

% % construct the HR and LR training pairs from the FEI face database
% [YH YL] = Training_LH(upscale,nTraining);

% for TestImgIndex = 1:nTesting
%     fprintf('\nProcessing  %d_test.jpg\n', TestImgIndex);

%     % read ground truth of one test face 
%     strh    = strcat('.\testFaces\',num2str(TestImgIndex),'_test.jpg');
%     im_h    = double(imread(strh));

%     % generate the input LR face by smooth and down-sampleing
%     psf         = fspecial('average', [4 4]); 
%     im_s    = imfilter(im_h,psf);
%     im_l    = imresize(im_s,1/upscale,'bicubic');
%     % upscale the LR face to HR size
%     im_b = imresize(im_l,upscale,'bicubic');
    
%     tic;
%     % hallucinate the high frequency face via TLcR
%     [im_SR] = TLcR_RL(im_b,YH,YL,upscale,patch_size,overlap,stepsize,window,tau,K,c);     
%     % add the high frequency face to result
%     [im_SR] = im_SR+im_b;
%     cputime(TestImgIndex) = toc;
%     imwrite(uint8(im_SR),strcat('./results/',num2str(TestImgIndex),'_',num2str(tau),'_TLcR.bmp'),'bmp');  

%     % compute PSNR and SSIM for Bicubi,c and TLcR method
%     bicubic_psnr(TestImgIndex) = psnr(im_b,im_h);
%     bicubic_ssim(TestImgIndex) = ssim(im_b,im_h);
%     TLcR_psnr(TestImgIndex) = psnr(im_SR,im_h);
%     TLcR_ssim(TestImgIndex) = ssim(im_SR,im_h);  


%     % % updata the result by reproducing learning
%     % for ls = 1:layer
%     %     im_lSR  = imfilter(im_SR,psf);
%     %     im_lSR  = imresize(im_lSR,1/upscale,'bicubic');    
%     %     im_lSR  = imresize(im_lSR,size(im_SR));
%     %     [im_SR] = TLcR_RL(im_b,cat(3,YH,im_SR-im_lSR),cat(3,YL,im_lSR),upscale,patch_size,overlap,stepsize,window,tau,K,c);
%     %     [im_SR] = im_SR+im_b;
%     %     % compute PSNR and SSIM for Bicubic and TLcR-RL method
%     %     TLcRRL_psnr(ls,TestImgIndex) = psnr(im_SR,im_h);
%     %     TLcRRL_ssim(ls,TestImgIndex) = ssim(im_SR,im_h);          
%     % end 
%     % imwrite(uint8(im_SR),strcat('./results/',num2str(TestImgIndex),'_',num2str(tau),'_TLcRRL.bmp'),'bmp');  
    
%     % display the objective results (PSNR and SSIM)
%     fprintf('PSNR for Bicubic:  %f dB\n', bicubic_psnr(TestImgIndex));
%     fprintf('PSNR for TLcR:     %f dB\n', TLcR_psnr(TestImgIndex));
%     %fprintf('PSNR for TLcR-RL:  %f dB\n', TLcRRL_psnr(layer,TestImgIndex));
%     fprintf('SSIM for Bicubic:  %f dB\n', bicubic_ssim(TestImgIndex));
%     fprintf('SSIM for TLcR:     %f dB\n', TLcR_ssim(TestImgIndex));
%     %fprintf('SSIM for TLcR-RL:  %f dB\n', TLcRRL_ssim(layer,TestImgIndex));

% end


% fprintf('===============================================\n');
% fprintf('Average PSNR for Bicubic:  %f dB\n', sum(bicubic_psnr)/nTesting);
% fprintf('Average PSNR for TLcR:     %f dB\n', sum(TLcR_psnr)/nTesting);
% %fprintf('Average PSNR for TLcRRL:   %f dB\n', sum(TLcRRL_psnr(layer,:))/nTesting);
% fprintf('Average SSIM for Bicubic:  %f dB\n', sum(bicubic_ssim)/nTesting);
% fprintf('Average SSIM for TLcR:     %f dB\n', sum(TLcR_ssim)/nTesting);
% %fprintf('Average SSIM for TLcR:     %f dB\n', sum(TLcRRL_ssim(layer,:))/nTesting);
% fprintf('===============================================\n');