%% References
% A Robust and Efficient Approach to License Plate Detection
% A Fast Algorithm for License Plate Detection 
% Research of license plate recognition based on HSV Space
% A new method for license plate detection based on color and edge information of Lab space
% A MORPHOLOGICAL-BASED LICENSE PLATE LOCATION
% Mean Shift for Accurate License Plate Localization
% HSI color based vehicle license plate detection

%% Program Start
clear all;
close all;
clc;

%% Read Image
rgb_im = imread("car-license-plate-dataset\images\Cars358.png");  
rgb_im = im2uint8(rgb_im); % 8bit
[def_rows def_cols] = size(rgb_im);
% figure,imshow(rgb_im);title('Original Image');

%%
%--Car Plate Detection--
%% RGB to HSI
rgb_im_normalized = double(rgb_im) / 255; % Represent the RGB image in [0 1] range
R = rgb_im_normalized(:,:,1);
G = rgb_im_normalized(:,:,2);
B = rgb_im_normalized(:,:,3);
 
% Converting the image to HSV to obtain the Hue and Saturation Channels
hsv_im = rgb2hsv(rgb_im);
H = hsv_im(:,:,1);
S = hsv_im(:,:,2);
I = sum(rgb_im_normalized, 3)./3;
 
% Creating the HSL Image
hsi_im = zeros(size(rgb_im));
hsv_im(:,:,1) = H;
hsv_im(:,:,2) = S;
hsv_im(:,:,3) = I;
% figure,imshow(I);title('I channel');

%% Sobel Vertical Edge Detection
s_v = [-1 0 1;-2 0 2;-1 0 1]; 
edge_im = imfilter(I, s_v, 'replicate');
edge_im = edge_im.^2; % Enhance Edges
% figure,imshow(edge_im);title('Sobel Vertical Edge');

%% Gaussian Blur
blur_im = imgaussfilt(edge_im, 0.8); % Enhance Intensity
% figure,imshow(blur_im);title('Guassian Blurred');

%% Otsu Thresholding
bin_im = imbinarize(blur_im);
% figure,imshow(bin_im);title('Binarized');

%% Fill Holes
bin_im = imfill(bin_im, "holes");
% figure,imshow(bin_im);title('Filled Holes');

%% Border Removal
margin = 40;
border_im = imcrop(bin_im, [margin, margin, def_cols - 2 * margin, def_rows - 2 * margin]); % crop margin*4. 
border_im = imclearborder(border_im);
border_im = logical(padarray(border_im, [margin margin], 0)); % Pad 0 
% figure,imshow(border_im);title('Border Removed');

%% Remove Minor Pixels 1
border_im = bwareaopen(border_im, 100);
% figure,imshow(border_im);title('Noise Removal');

%% Histogram Analysis Horizontal
row_sum = sum(border_im, 2); % Edge Horizontal Histogram
thresh_hist = 0.6; % error rate
cRows = find(row_sum > (thresh_hist*max(row_sum))); % candidate row

%% Masking Candidate Region
msk_h = zeros(size(border_im));
msk_h(cRows,:) = 1; % define a kernel                
candidate_row = msk_h.*border_im;
candidate_row = imbinarize(candidate_row);
% figure;imshow(candidate_row);title('Candidate Rows');

%% Remove Minor Pixels 2
candidate_row = bwareaopen(candidate_row, 50);
% figure,imshow(candidate_row);title('Noise Removal 2');

%% Morphology (Dilation - Vertical)
Dy    = strel('rectangle',[100,4]);      
MBy   = imdilate(candidate_row,Dy);           
MBy   = imfill(MBy,'holes');          
% figure();imshow(MBy);title('Vertical Dilation');

%% Morphology (Dilation - Horizontal)
Dx    = strel('rectangle',[4,100]);     
MBx   = imdilate(candidate_row,Dx);          
MBx   = imfill(MBx,'holes');         
% figure();imshow(MBx);title('Horizontal Dilation');

%% Dilation Intersection
dil_im   = MBx.*MBy; % Joint Dilation                     
dil_im = imfill(dil_im, 'holes');
% figure();imshow(dil_im);title('Dilation Intersect');

%% Morphology (Close)
Cr = strel('rectangle', [40 40]);
closed_im = imclose(dil_im, Cr);
closed_im = imbinarize(closed_im);
% figure,imshow(closed_im);title('Closed Morp');

%% Remove Pixel not in range
area_filt_im = bwareafilt(closed_im, [500 25000]); 
% figure,imshow(area_filt_im);title('Area Filtered');

%% Erosion
Er = strel('rectangle', [9 9]);
er_im = imerode(area_filt_im, Er);
% figure,imshow(er_im);title('Erosion');

%% Morphology (Square - large)
Ds_xl = strel('rectangle', [65 50]);  %65, 50 
squared_im_large   = imdilate(er_im, Ds_xl);
% figure,imshow(squared_im_large);title('Dilated Sqaure Large');

%% Morphology (Square - small)
Ds_xs = strel('square', 20);    
squared_im_small   = imdilate(area_filt_im, Ds_xs);
% figure,imshow(squared_im_small);title('Dilated Sqaure Small');

figure,
subplot(431);imshow(rgb_im);title('Original Image');
subplot(432);imshow(I);title('I Channel');
subplot(433);imshow(edge_im);title('Edge detected image');
subplot(434);imshow(blur_im);title('Gaussian blurred');
subplot(435);imshow(candidate_row);title('Histogram Analysis');
subplot(436);imshow(squared_im_small);title('Morphological Square Small');
subplot(437);imshow(squared_im_large);title('Morphological Square Large');

%% CCL Filter
ccl_bb_box_xl = bb_generator(squared_im_large);
ccl_bb_box_xs = bb_generator(squared_im_small);
ccl_bb_box_xl = car_plate_ccl_filter(ccl_bb_box_xl, "xl");
ccl_bb_box_xs = car_plate_ccl_filter(ccl_bb_box_xs, "xs");

ccl_bb_boxes = ccl_bb_box_xl;
for i = 1:numel(ccl_bb_box_xs)
    ccl_bb_boxes{length(ccl_bb_boxes)+1} = ccl_bb_box_xs{i};
end

%% Remove overlapping
figure,
sgtitle('CCL filtered & Removed Overlapping');
subplot(121);
print_bb_box(area_filt_im, rgb_im, ccl_bb_boxes); % before removing overlapping
ccl_bb_boxes = removed_overlapped_bb(ccl_bb_boxes);
subplot(122);
print_bb_box(area_filt_im, rgb_im, ccl_bb_boxes); % after removing overlapping

%% Crop Candidate Region
cropped_im = cell(length(ccl_bb_boxes),1);
for i = 1 : length(ccl_bb_boxes)
    cropped_im{i} = imcrop(rgb_im, ccl_bb_boxes{i}); 
end

%%
%--Car Plate Character Segmentation--
%% 
char_seg_im = cropped_im;
plates_rgb_cropped = char_seg_im;
plates_region_bb = ccl_bb_boxes;
plates_characters_region = {};
plates_characters_bb = {};
plates_counter = 1;
for i = 1:numel(char_seg_im)
    %% Convert Cropped Images to Gray Scale
    cur_cand_rgb = char_seg_im{i};
    cur_cand = rgb2gray(char_seg_im{i});
    [cand_rows, cand_cols] = size(cur_cand);
    
    %% Binarize Image
    cur_cand = imbinarize(cur_cand);
    bin_cur_cand = cur_cand;
    % figure,imshow(cur_cand);title('Binarized Candidate');

    %% First Inversion (Inverse white to black, and black to white pixel)
    whitepx = sum(cur_cand(:));
    totalpx = length(cur_cand(:));
    ratio = whitepx/totalpx;
    % disp(ratio);
    if(ratio > 0.5 || (ratio > 0.36 && ratio < 0.37) || (ratio > 0.45 && ratio < 0.46) )
        cur_cand = ~cur_cand;
        % figure,imshow(cur_cand);title('Inversed Candidate');
    end

    %% Clear Border
    margin_x = round(cand_cols*0.01);    % 1% left/right
    margin_y = 1;
    cur_cand = imcrop(cur_cand, [margin_x, margin_y, cand_cols - 2 * margin_x, cand_rows - 2 * margin_y]);
    cur_cand = imclearborder(cur_cand);
    cur_cand = logical(padarray(cur_cand, [margin_x margin_y], 0)); % Pad 0 
    % figure,imshow(cur_cand);title('Cleared Border Candidate');

    %% Remove Minor Pixels 3
    backup_cur_cand = cur_cand;
    cur_cand = bwareaopen(cur_cand, 35); 
    if sum(cur_cand) == 0
        cur_cand = bwareaopen(backup_cur_cand, 20);
    end
    % figure,imshow(cur_cand);title('Noise Removal 3');

    %% Second Inversion
    second_inversion_flag = 0;
    whitepx = sum(cur_cand(:));
    totalpx = length(cur_cand(:));
    ratio = whitepx/totalpx;
    if(ratio > 0.25) 
        cur_cand = ~cur_cand;
        second_inversion_flag = 1;
        % figure,imshow(cur_cand);title('Second Inversed Candidate');
    end

    %% Clear Border & Remove Minor Pixels 4
    if second_inversion_flag == 1
        margin_x = round(cand_cols*0.02);       % 2% left/right
        % margin_y = round(cand_rows*0.02);     % 2% up/down
        margin_y = 1;
        cur_cand = imcrop(cur_cand, [margin_x, margin_y, cand_cols - 2 * margin_x, cand_rows - 2 * margin_y]);
        cur_cand = imclearborder(cur_cand);
        cur_cand = logical(padarray(cur_cand, [margin_x margin_y], 0)); % Pad 0
        cur_cand = bwareaopen(cur_cand, 10); 
        % figure,imshow(cur_cand);title('Cleared Border Candidate');
    end
    
    %% Jump to next candidate if no white pixels
    if sum(cur_cand) == 0
        plates_region_bb{i} = []; % Remove current bounding box 
        plates_rgb_cropped{i} = [];
        continue;
    end

    %% CCL Filter
    ccl_bb_box_char = bb_generator(cur_cand);
    ccl_bb_box_char = character_ccl_filtering(ccl_bb_box_char);

    no_bounding_box_flag = 0;

    if length(ccl_bb_box_char) < 4 % last inverse as final attempt to find character
        no_bounding_box_flag = 1;
        cur_cand = ~cur_cand;
        cur_cand = imclearborder(cur_cand);
        cur_cand = bwareaopen(cur_cand, 35); % remove noise
        % figure,imshow(cur_cand);title('Third Inverse Candidate');

        ccl_bb_box_char = bb_generator(cur_cand);
        ccl_bb_box_char = character_ccl_filtering(ccl_bb_box_char);
    end

    %% Jump to next candidate if less than 4 bounding box 
     if length(ccl_bb_box_char) < 4 
         plates_region_bb{i} = []; % Remove current bounding box 
         plates_rgb_cropped{i} = [];
         continue;
     end

    %% Store Valid Candidate For OCR
    % Remove Empty Cell
    plates_characters_region{plates_counter} = cur_cand;
    plates_characters_bb{plates_counter} = ccl_bb_box_char;
    plates_counter = plates_counter + 1;
end

%%
%--OCR--
%%
plats_ocr_results = {};

if numel(plates_characters_region) == 0
    plats_ocr_results{1} = "Not Detected";
    % disp(plats_ocr_results{1});
end

for i = 1: numel(plates_characters_region)
    character_bb_boxes = plates_characters_bb{i};

    plate_char = [];
    for k = 1:numel(character_bb_boxes)
        %% Preprocessing
        current_plate_region = bwmorph(plates_characters_region{i},'thin', 1); % Character Thinning
        current_plate_region = ~current_plate_region; % Complement for black character

        %% Cropping
        temp = imcrop(current_plate_region, character_bb_boxes{k});
        temp = logical(padarray(temp, [5 5], 1));
        % figure,imshow(temp);title('Thinned Character');
        [temp_height, temp_width] = size(temp);

        %% Resizing
        temp = imresize(temp, [temp_height*1.2 temp_width*1.2]);

        %% Execute OCR
        ocrResult = ocr(temp, 'TextLayout', 'Block', 'CharacterSet', 'ABCDEFGHIJKLMNOPQRSTUVWXYZa01234567890-');
        plate_char = [plate_char ocrResult.Text]; 
    end
    
    if ~isempty(plate_char)
        plate_char = erase(plate_char, newline); % Add character as same line
    else
        plate_char = "Characters Not Detected";
    end
    plats_ocr_results{i} = plate_char;
end

%%
%--Displaying Results--
%%
plates_region_bb = plates_region_bb(~cellfun('isempty', plates_region_bb)); % Remove empty cells
if ~isempty(plates_region_bb)
    figure,print_bb_box(rgb_im, rgb_im, plates_region_bb);
else
    figure,print_bb_box(rgb_im, rgb_im, ccl_bb_boxes);
    print_rbg_img_with_ocr({rgb_im}, {"Character segmentation failed"});
    print_license_plate_detected(char_seg_im);
end
for i=1:numel(plates_characters_region)
    % print_bb_box(plates_characters_region{i}, plates_characters_region{i}, plates_characters_bb{i});
end
plates_rgb_cropped = plates_rgb_cropped(~cellfun('isempty', plates_rgb_cropped)); % Remove empty cells
% print_license_plate_detected(plates_rgb_cropped);
print_rbg_img_with_ocr(plates_rgb_cropped, plats_ocr_results);

%% Functions
function print_license_plate_detected(imgs)
    for i = 1:numel(imgs)
        figure,
        imshow(imgs{i});
        title('License Plate Detected');
    end
end

function print_rbg_img_with_ocr(rgb_imgs, ocr)
    for i = 1:numel(rgb_imgs)
        figure,
        imshow(rgb_imgs{i});
        title('OCR Result');
        hold on;

        annotation('textbox', [0.432 0.0 0.8 0.1], ...
            'String', ['Recognized License Plate: ' ocr{i}], ...
            'Color', [1 0 0], ...
            'FontWeight', 'bold', ...
            'EdgeColor', 'none')
        hold off;
    end
end

function print_bb_box(img, rgb_img, bb_box)
    % figure,
    imshow(rgb_img);
    title('Bounding Regions');
    hold on
    for i = 1 : length(bb_box)
        rectangle('Position', bb_box{i}, 'EdgeColor', 'r', 'LineWidth', 2);
    end
    hold off
end

function bb_box = bb_generator(img)
    [L, num] = bwlabel(img, 8);
    box = regionprops(L, 'BoundingBox');
   
    bb_box = cell(1,length(box));
    
    for i = 1 : length(box)
        cur_bb_box = box(i).BoundingBox;
        bb_box{i} = cur_bb_box;
    end
end


function valid_bb_box = car_plate_ccl_filter(boundingbox, type)
    valid_bb_box = cell(length(boundingbox),1);

    for i = 1:numel(boundingbox)
        w = boundingbox{i}(3);
        h = boundingbox{i}(4);
    
        if h >= w 
            valid_bb_box{i} = [];
            continue
        end
    
        if w/(w+h) <= 0.6 % width ratio
            valid_bb_box{i} = [];
            continue
        end
    
        if w < 50 && h < 50 
            valid_bb_box{i} = [];
            continue
        end
    
        valid_bb_box{i} = boundingbox{i};
    end
    % Remove Empty Cell
    valid_bb_box = valid_bb_box(~cellfun('isempty', valid_bb_box));
end


function is_character = character_ccl_filtering(boundingbox) 
    is_character = cell(length(boundingbox),1);

    for i = 1:numel(boundingbox)
        w = boundingbox{i}(3);
        h = boundingbox{i}(4);
 
        if w/(h+w) > 0.5
            is_character{i} = [];
            continue;
        end

        if w/h < 0.4
            is_character{i} = [];
            continue;
        end

        is_character{i} = boundingbox{i};
    end
    % Remove Empty Cell
    is_character = is_character(~cellfun('isempty', is_character));
end


function bb_boxes_result = removed_overlapped_bb(bb_boxes)
    for i = 1:numel(bb_boxes)-1
        cur_bb_box = bb_boxes{i};
        if isempty(cur_bb_box) == 1 % slot empty jump to next slot
           continue;
        end
        area1 = cur_bb_box(3) * cur_bb_box(4);        
        for j = i+1:numel(bb_boxes)
            checking_bb_box = bb_boxes{j};
            if isempty(checking_bb_box) == 1 % slot empty jump to next slot
                continue;
            end
            area2 = checking_bb_box(3)  * checking_bb_box(4);
            overlapRatioA = bboxOverlapRatio(cur_bb_box, checking_bb_box, "Min");
            overlapRatioB = bboxOverlapRatio(cur_bb_box, checking_bb_box);
            % disp(overlapRatioB);
            if overlapRatioB > 0
                if area1 > area2
                    bb_boxes{j} = [];
                else
                    bb_boxes{i} = [];
                end
            end
        end
    end
    % Remove Empty Cell
    bb_boxes = bb_boxes(~cellfun('isempty', bb_boxes));
    bb_boxes_result = bb_boxes;
end
