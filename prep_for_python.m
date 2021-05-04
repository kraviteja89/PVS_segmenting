clear all
close all
clc
%%  File details - change values here
data_location = 'G:/CAG-GFP/3D_data/';
animal_ID =  'RK050';
day_ID = '210425';
file_num = '002';
n_steps = 10;
st_frames =6:10;
slice_depths = linspace(80,40,n_steps);
slice_depths = slice_depths(st_frames);
frame_size = 224;
%% read image data
f_name = [data_location animal_ID '/Green/' day_ID '_' file_num];
f_name1 = [data_location animal_ID '/Red/' day_ID '_' file_num];

f_names = dir([f_name '*.TIF']);
f_names1 = dir([f_name1 '*.TIF']); % sometimes MView writes the output in two Tif files
the_info = imfinfo([f_names(1).folder '/'  f_names(1).name]);
text_hold=textscan(the_info(1).ImageDescription,'%s','delimiter','\n');
frame_dur = str2double(text_hold{1}{24}(17:end-3));
Fs = 1/frame_dur;
n_frames = size(the_info,1);

if length(f_names)>1
    the_info2 = imfinfo([f_names(2).folder '/'  f_names(2).name]);
    n_frames2 = size(the_info2,1);
end
for n2 = 1:length(st_frames)
    st_frame=st_frames(n2);
    for n1 = st_frame:n_steps:n_frames
        the_data_g(:,:,n2,1+(n1-st_frame)/n_steps) = double(imread([f_names(1).folder '/'  f_names(1).name],n1));
        the_data_r(:,:,n2,1+(n1-st_frame)/n_steps) = double(imread([f_names1(1).folder '/'  f_names1(1).name],n1));
    end
    if length(f_names)>1
        for n = n1 + n_steps: n_steps : n_frames + n_frames2
            the_data_g(:,:,n2,1+(n-st_frame)/n_steps) = double(imread([f_names(2).folder '/'  f_names(2).name],n - n_frames));
            the_data_r(:,:,n2,1+(n-st_frame)/n_steps) = double(imread([f_names1(2).folder '/'  f_names1(2).name],n - n_frames));
        end
    end
end
[ht, wd, n_slices,n_frames] = size(the_data_g);
%% Read running data
walking_data = dlmread([data_location animal_ID '/' day_ID '_' file_num '.txt']);
global acc_cutoff
acc_cutoff= 1e-4;
walking_bin = velocity_proc2(walking_data(:,2),1000,Fs);
walking_bin = double(conv(walking_bin, ones(1,5), 'same')>0);
walking_bin = downsample(walking_bin, n_steps);
walking_bin = double(conv(walking_bin, ones(1,5), 'same')>0);
%% match lengths of walking data and image data
try
    walking_bin = walking_bin(1:n_frames);
catch
    n_frames = length(walking_bin);
    the_data_g = the_data_g(:,:,:,1:n_frames);
    the_data_r = the_data_r(:,:,:,1:n_frames);
end
%% Find rest period for baseline
run_frames = [0; find(walking_bin)];
run_diffs = diff(run_frames);
[max_rest,max_rest_ind] = max(run_diffs);
if max_rest < 10
    rest_period = [1 10];
else
    rest_period = [run_frames(max_rest_ind) + 5, run_frames(max_rest_ind+1)-1];
end

%% select ROIs
lims =[ht/2 - frame_size/2, wd/2 - frame_size/2, frame_size, frame_size];
lims = repmat(lims, n_slices, 1);
the_data_g1 = zeros(frame_size, frame_size, n_slices, n_frames);
the_data_r1 = zeros(frame_size, frame_size, n_slices, n_frames);
for n2 = 1:n_slices
    base_image_g = median(squeeze(the_data_g(:,:,n2,rest_period(1):rest_period(2))),3);
    base_image_r = median(squeeze(the_data_r(:,:,n2,rest_period(1):rest_period(2))),3);
    
    base_image = zeros([size(base_image_r),3]);
    base_image(:,:,1) = base_image_r/prctile(base_image_r(:), 99);
    base_image(:,:,2) = base_image_g/prctile(base_image_g(:), 99);
    base_image(:,:,3) = base_image_r/prctile(base_image_r(:), 99);
    
    fig = figure(100);
    set(fig, 'Position', [50 50 1000 900])
    imshow(base_image, [])
    axis equal
    axis off
    title(['Slice :' num2str(n2) ' Adjust the ROI and enter y in the command window'])
    adjusted = 'n';
    if n2==1
        h = drawrectangle('Position',lims(n2,:) , 'InteractionsAllowed', 'translate');
    else
        h = drawrectangle('Position',lims(n2-1,:) , 'InteractionsAllowed', 'translate');
    end
    while ~strcmpi(adjusted, 'y')
        pause(5)
        adjusted = input('Is the ROI positioned correctly (y/n): ', 's');
    end
    lims(n2,:) = h.Position;
    lims = floor(lims);
    the_data_g1(:,:,n2,:) = the_data_g(lims(n2,2):lims(n2,2)+lims(n2,4)-1,lims(n2,1):lims(n2,1)+lims(n2,3)-1,n2,:);
    the_data_r1(:,:,n2,:) = the_data_r(lims(n2,2):lims(n2,2)+lims(n2,4)-1,lims(n2,1):lims(n2,1)+lims(n2,3)-1,n2,:);
end

the_data_g = the_data_g1;
the_data_r = the_data_r1;

ht = frame_size; wd = frame_size;

%% rescale pixel intensities so that 1% of pixels are saturated
for n2 = 1:length(st_frames)
    the_data_g(:,:,n2,:) = medfilt3(squeeze(the_data_g(:,:,n2,:)), [5 5 3]);
    the_data_r(:,:,n2,:) = medfilt3(squeeze(the_data_r(:,:,n2,:)), [5 5 3]);
    for n = 1:n_frames
        green_frame = squeeze(the_data_g(:,:,n2,n));
        red_frame = squeeze(the_data_r(:,:,n2,n));
        
        % remove noise from red channel
        red_frame = red_frame - prctile(red_frame(:),50);
        red_frame(red_frame<0) = 0;
        
        green_frame =  green_frame/prctile(green_frame(:),99);
        red_frame =  red_frame/prctile(red_frame(:),99);
        
        green_frame(green_frame>1) = 1;
        red_frame(red_frame>1) = 1;
        
        the_data_g(:,:,n2,n) = green_frame;
        the_data_r(:,:,n2,n) = red_frame;
    end
end
%% Create test data
n_samples = 3;
fig = figure(101);
set(fig, 'Position', [50 50 1400 900])

still_frames = find(~walking_bin);
run_frames = find(walking_bin);
still_samples = still_frames(randi(length(still_frames),1,n_samples));
run_samples = run_frames(randi(length(run_frames),1,n_samples));

sample_frames = [still_samples; run_samples];
n_samples = n_samples*2;

image_samples_g = the_data_g(:,:,:,sample_frames);
image_samples_r = the_data_r(:,:,:,sample_frames);
ROI_samples = zeros(ht,wd,n_slices,n_samples);


for n2 =1:n_slices
    for n = 1:n_samples
        green_frame = squeeze(the_data_g(:,:,n2,sample_frames(n)));
        red_frame = squeeze(the_data_r(:,:,n2,sample_frames(n)));
        
        full_clr_sample = zeros(ht,wd,3);
        full_clr_sample(:,:,1) = red_frame;
        full_clr_sample(:,:,2) =  green_frame;
        full_clr_sample(:,:,3) = red_frame;
        
        clf(fig)
        image(full_clr_sample)
        axis equal
        axis off
        title(['Slice:' num2str(n2) ', frame:' num2str(n) ' - Draw PVS boundary'])
        h = drawpolygon;
        ROI_samples(:,:,n2,n) = h.createMask;
    end
end
%% Save data
save(['datasets/' animal_ID '_' day_ID '_' file_num '_python.mat'], 'the_data_g', 'the_data_r', 'Fs',...
    'slice_depths', 'n_steps', 'frame_size', 'lims', 'image_samples_g', 'image_samples_r', 'ROI_samples',...
    'sample_frames', 'rest_period', 'walking_bin','-v7.3')

