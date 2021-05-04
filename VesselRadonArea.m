function [vessel_area, vessel_ROI] = VesselRadonArea(current_frame, rtd_threshold)
[ht, wd] = size(current_frame);


the_angles=1:1:180;
if nargin<3
    rtd_threshold=.25;%0.5 'half max'
end
irtd_threhold=.2;%0.2 gets rid of the noise from the inverse radon
current_frame=current_frame-mean(current_frame(:));
radon_hold_image=radon(current_frame,the_angles);

% Radon cleanup
for k=1:length(the_angles)
    % normalize so that for each angle, the transfomed image is between
    % 0 and 1
    radon_hold_image(:,k)=radon_hold_image(:,k)-min(radon_hold_image(:,k));
    radon_hold_image(:,k)=radon_hold_image(:,k)/max(radon_hold_image(:,k));
    % find the peak of the projection at this angle
    [maxpoint(k),maxpointlocation(k)]=max(radon_hold_image(:,k));
    % threshold at half maximum
    try
        [~,min_edge(k)]=max(find(radon_hold_image(1:maxpointlocation(k),k)<rtd_threshold));
        [~,max_edge(k)]=max(find(radon_hold_image(maxpointlocation(k)+1:end,k)>rtd_threshold));
    catch
        min_edge(k)=0;
        max_edge(k)=0;
    end
    
    radon_hold_image(1:min_edge(k),k)=0;
    radon_hold_image((maxpointlocation(k)+max_edge(k)):end,k)=0;
end

irtd_norm=iradon(double(radon_hold_image>rtd_threshold*max(radon_hold_image(:))),(the_angles),'linear','Hamming');


[cc,l] = bwboundaries(irtd_norm>irtd_threhold*max(irtd_norm(:)),4);
numPixels = cellfun(@length,cc);
[~,idx] = max(numPixels);
area_filled=regionprops(l,'FilledArea','PixelList');
if isempty(area_filled)
    vessel_area = 0;
    vessel_ROI = zeros(ht, wd);
    wall_ROI = zeros(ht, wd);
    return
end
vessel_area = area_filled(idx).FilledArea;
vessel_pixels = cc{idx};
vessel_ROI =poly2mask(vessel_pixels(:,2),vessel_pixels(:,1), ht,wd);
end