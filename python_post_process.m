clear all
close all
clc
%% Data and model parameters
animal_ID =  'RK050';
day_ID = '210425';
file_num = '002';
model = 'vgg16';
last_layer = 'pool4';
make_3D_video = true;
st_time = 0;
%% Load the data
load(['datasets/' animal_ID '_' day_ID '_' file_num '_python.mat'])
try
     PVS_ROIs = double(h5read(['datasets/' animal_ID '_' day_ID '_' file_num '_' model '_' last_layer '.h5'], '/PVS_ROIs'));
     PVS_areas = double(h5read(['datasets/' animal_ID '_' day_ID '_' file_num '_' model '_' last_layer '.h5'], '/PVS_areas'));
catch
     fprintf('Python h5 file %s not found: exiting', ['datasets/' animal_ID '_' day_ID '_' file_num '_' model '_' last_layer '.h5']);
     return
end

%% Calculate vessel radii - using radon cleanup
[ht, wd, n_slices, n_frames] = size(the_data_g);
vessel_ROIs = zeros(size(the_data_g));
vessel_areas = zeros(n_slices, n_frames);
vessel_centroids = zeros(2,n_slices, n_frames);

[b,a] = butter(3,2/3);

for n2 = 1:n_slices
    for n = 1:n_frames
        red_frame = squeeze(the_data_r(:,:,n2,n));
        red_frame(PVS_ROIs(:,:,n2,n)>0) = 0;% remove perivascular macrophages
        [vessel_areas(n2,n), vessel_ROIs(:,:,n2,n)] = VesselRadonArea(red_frame, 0.1);
        vessel_centroids(:,n2,n) = ROI2centroid(vessel_ROIs(:,:,n2,n));
    end
    vessel_areas(n2,:) = filtfilt(b,a,vessel_areas(n2,:));
    vessel_centroids(1,n2,:) = filtfilt(b,a, squeeze(vessel_centroids(1,n2,:)));
    vessel_centroids(2,n2,:) = filtfilt(b,a, squeeze(vessel_centroids(2,n2,:)));    
end        
%% Calculate vessel centroid displacement
vessel_centroids_rest = median(vessel_centroids(:,:,rest_period(1):rest_period(2)), 3);
vessel_offset = bsxfun(@minus, vessel_centroids_rest, vessel_centroids_rest(:,1));
vessel_displacement = bsxfun(@minus, vessel_centroids, vessel_centroids_rest);
%% Calculate the PVS perimeters
PVS_perimeters = zeros(64, 2, n_slices, n_frames);
for n2 = 1:n_slices
    for n = 1:n_frames
        [cc,~] = bwboundaries(squeeze(PVS_ROIs(:,:,n2,n))>0,4);
        try
            pos = cc{1};
            PVS_perimeters(:,:,n2,n) = interppolygon(pos,64);
        catch
            continue % for the case of zero PVS area
        end
    end
end
%%


%% Make a 3D plot
if make_3D_video
    walk_bin = double(walking_bin);
    walk_bin(~walk_bin) = nan;
    pixel_scale = 0.82/3;
    [X,Y,Z] = meshgrid((1:wd)*pixel_scale, (1:ht)*pixel_scale, slice_depths);
    
    fig = figure;%(666);
    set(fig, 'Position',[50 50 1400 900])
    outputVideo = VideoWriter(['videos/' animal_ID '_' day_ID '_' file_num  '_3Dpy_results.mp4'], 'MPEG-4');
    outputVideo.FrameRate = 10;
    open(outputVideo);
    
    t_vec = (1:n_frames)/(Fs/n_steps);
    for n2 = 1:n_slices
        subplot(4,3,8:9)
        PVS_area = double(PVS_areas(n2,:));
        PVS_area_change = PVS_area/mean(PVS_area(rest_period(1):rest_period(2))) -1;
        plot(t_vec, PVS_area_change*100, 'LineWidth',2)
        hold on
        
        subplot(4,3,11:12)
        vessel_area = vessel_areas(n2,:);
        vessel_area_change = vessel_area/mean(vessel_area(rest_period(1):rest_period(2))) -1;
        plot(t_vec, vessel_area_change*100, 'LineWidth',2)
        hold on
        if n2 == 1
            legs{1} = ['z = ' num2str(round(slice_depths(n2),2)) '\mum'];
        else
            legs{n2} = [num2str(round(slice_depths(n2),2)) '\mum'];
        end
    end
    
    legs{n2+1} = 'Running';
    subplot(4,3,8:9)
    xlim(st_time + [0 100])
    % xlabel('time(s)')
    ylabel('\Delta PVS area %')
    yyaxis right
    plot(t_vec, walk_bin, 'k', 'LineWidth', 4)
    yticks([])
    ylim([0 1.2])
    set(gca, 'Box', 'off')
    hlegend = legend(legs, 'Location', 'northoutside', 'Orientation', 'horizontal');
    hold off
    set(gca, 'FontSize', 12)
    
    
    subplot(4,3,11:12)
    xlim(st_time + [0 100])
    xlabel('time(s)')
    ylabel('\Delta vessel area %')
    yyaxis right
    plot(t_vec, walk_bin, 'k', 'LineWidth',4)
    yticks([])
    ylim([0 1.2])
    set(gca, 'Box', 'off')
    % legend(legs)
    hold off
    set(gca, 'FontSize', 12)
    
    blank_frame = zeros(ht, wd);
    set(gcf, 'Color', 'w')
    
    for n =ceil(st_time*Fs/n_steps) + (1:100*Fs/n_steps)
        curr_offset = vessel_offset + vessel_displacement(:,:,n);
        vessel_block =  displace_ROI(vessel_ROIs(:,:,:,n), curr_offset);
        PVS_block = displace_ROI(PVS_ROIs(:,:,:,n), curr_offset);
        
        subplot(4,3,[7 10])
        cla
        p = patch(isosurface(X,Y,Z,  vessel_block, 0.5));
        p.EdgeColor = 'none';
        p.FaceColor = [1 0 1];
        p.FaceAlpha = 0.5;
        hold on
        p = patch(isosurface(X,Y,Z,  PVS_block, 0.5));
        p.EdgeColor = 'none';
        p.FaceColor = [0 1 1];
        p.FaceAlpha = 0.5;
        view(3)
        daspect([1 1 0.1])
        set(gca, 'ZDir', 'reverse')
        
        for n2 = [1 ceil(n_slices/2) n_slices]
            green_frame = the_data_g(:,:,n2,n);
            D = zeros(ht, wd, 2);
            D(:,:,1) = curr_offset(1,n2);
            D(:,:,2) = curr_offset(2,n2);
            green_frame = imwarp(green_frame, D);
            h = surface([1 wd ;1 wd]*pixel_scale, [1 1 ;ht ht]*pixel_scale, slice_depths(n2)*ones(2), 'CData', cat(3,blank_frame,green_frame,blank_frame),'FaceColor', 'texturemap', 'EdgeColor', 'none' );
            alpha(h, 0.5)
        end
        hold off
        set(gca, 'View', [42  28])
        % xlabel('x(\mum)')
        % ylabel('y(\mum)')
        zlabel('z(\mum)')
        set(gca, 'FontSize', 20)
        title(['t = ' num2str(round(t_vec(n),1)) 's'])
        
        top_green = the_data_g(:,:,1,n);
        top_red = the_data_r(:,:,1,n);
        top_image = zeros(ht, wd,3);
        top_image(:,:,1) = top_red;
        top_image(:,:,2) = top_green;
        top_image(:,:,3) = top_red;
        top_image = displace_ROI(top_image, repmat(curr_offset(:,1), 1, 3));
        
        
        subplot(4,2,[1 3])
        imshow(top_image)
        hold on
        plot(PVS_perimeters(:,2,1,n) - curr_offset(1,1), PVS_perimeters(:,1,1,n) - curr_offset(2,1), 'w', 'LineWidth',2)
        hold off
        title([num2str(slice_depths(1)) '\mum from brain surface'])
        
        bot_green = the_data_g(:,:,n_slices,n);
        bot_red = the_data_r(:,:,n_slices,n);
        bot_image = zeros(ht, wd,3);
        bot_image(:,:,1) = bot_red;
        bot_image(:,:,2) = bot_green;
        bot_image(:,:,3) = bot_red;
        bot_image = displace_ROI(bot_image, repmat(curr_offset(:,n_slices), 1, 3));
        
        
        subplot(4,2,[2 4])
        h = imshow(bot_image);
        hold on
        plot(PVS_perimeters(:,2,n_slices,n) - curr_offset(1,n_slices), PVS_perimeters(:,1,n_slices,n) - curr_offset(2,n_slices), 'w', 'LineWidth',2)
        hold off
        title([num2str(slice_depths(n_slices)) '\mum from brain surface'])
        currentFrame = getframe(gcf);
        writeVideo(outputVideo, currentFrame);
    end
    close(outputVideo)
end


