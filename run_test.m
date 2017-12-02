%% Initialize
clear

%% Set dimesion
N = 10^6;

%% Set source vectors and scalar values
src1 = ones(N, 1);
src2 = 2*ones(N, 1);
k1 = 2;
k2 = 3;

%% Measure elapsed time with MATLAB
tic;
dst1 = k1*src1 + k2*src2;
Elapsed1 = toc;

%% Measure elapsed time with MEX CUDA
tic;
d_src1 = gpuArray(src1);
d_src2 = gpuArray(src2);
d_dst2 = vec_add(d_src1, d_src2, k1, k2);
dst2 = gather(d_dst2);
Elapsed2 = toc;

%% Display result
disp(['N = ' num2str(N)]);
disp(['Elapsed time with MATLAB : ' num2str(Elapsed1) ' sec.']);
disp(['Elapsed time with MEX CUDA : ' num2str(Elapsed2) ' sec.']);