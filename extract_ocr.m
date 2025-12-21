% MATLAB OCR Script for Documentation Images
% This script extracts text from MATLAB documentation images using MATLAB's built-in OCR

function extract_ocr_from_docs()
    % Main function to extract OCR text from all documentation images

    % Configuration
    docs_root = './matlab_documents';
    output_file = 'image_ocr_data.json';

    % Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif'};

    % Find all image files in the documentation
    fprintf('🔍 Scanning for images in %s...\n', docs_root);

    image_files = {};
    for ext = image_extensions
        pattern = fullfile(docs_root, '**', ['*' ext{1}]);
        files = dir(pattern);
        for i = 1:length(files)
            % Get full path relative to docs_root
            rel_path = strrep(files(i).folder, [pwd filesep docs_root], '');
            if rel_path(1) == filesep
                rel_path = rel_path(2:end);
            end
            full_rel_path = fullfile(rel_path, files(i).name);

            % Skip very small files (likely icons)
            if files(i).bytes < 1024  % Less than 1KB
                continue;
            end

            image_files = [image_files; {fullfile(files(i).folder, files(i).name), full_rel_path}];
        end
    end

    fprintf('📊 Found %d image files to process\n', size(image_files, 1));

    % Process images in batches
    batch_size = 50;
    total_images = size(image_files, 1);
    ocr_results = {};

    for start_idx = 1:batch_size:total_images
        end_idx = min(start_idx + batch_size - 1, total_images);
        fprintf('🔄 Processing batch %d-%d of %d...\n', start_idx, end_idx, total_images);

        batch_files = image_files(start_idx:end_idx, :);
        batch_results = process_image_batch(batch_files);

        ocr_results = [ocr_results; batch_results];

        % Save intermediate results every 5 batches
        if mod(end_idx, 250) == 0 || end_idx == total_images
            save_results(ocr_results, output_file);
            fprintf('💾 Saved intermediate results (%d images processed)\n', length(ocr_results));
        end
    end

    % Final save
    save_results(ocr_results, output_file);
    fprintf('✅ OCR extraction complete! Processed %d images.\n', length(ocr_results));
end

function results = process_image_batch(image_files)
    % Process a batch of images with OCR

    results = {};

    for i = 1:size(image_files, 1)
        full_path = image_files{i, 1};
        rel_path = image_files{i, 2};

        try
            % Read the image
            img = imread(full_path);

            % Apply OCR
            ocr_result = ocr(img, 'TextLayout', 'Block');

            % Extract text and confidence
            extracted_text = strtrim(ocr_result.Text);
            confidence = ocr_result.CharacterConfidences;
            avg_confidence = mean(confidence(confidence > 0)); % Average of recognized characters

            % Only keep results with reasonable confidence and text
            if ~isempty(extracted_text) && length(extracted_text) > 10 && avg_confidence > 0.3
                result = struct(...
                    'path', rel_path, ...
                    'full_path', full_path, ...
                    'ocr_text', extracted_text, ...
                    'confidence', avg_confidence, ...
                    'word_count', length(strsplit(extracted_text)), ...
                    'character_count', length(extracted_text) ...
                );
                results = [results; {result}];
            end

        catch ME
            fprintf('⚠️  Error processing %s: %s\n', rel_path, ME.message);
            % Add error entry
            result = struct(...
                'path', rel_path, ...
                'full_path', full_path, ...
                'ocr_text', '', ...
                'confidence', 0, ...
                'error', ME.message, ...
                'word_count', 0, ...
                'character_count', 0 ...
            );
            results = [results; {result}];
        end

        % Progress indicator
        if mod(i, 10) == 0
            fprintf('  → Processed %d/%d images in batch\n', i, size(image_files, 1));
        end
    end
end

function save_results(results, filename)
    % Save OCR results to JSON file

    % Convert MATLAB structs to JSON-compatible format
    json_data = {};
    for i = 1:length(results)
        result = results{i};

        % Ensure all fields are JSON-compatible
        json_entry = struct();
        fields = fieldnames(result);
        for j = 1:length(fields)
            field_name = fields{j};
            value = result.(field_name);

            % Convert MATLAB types to JSON-compatible types
            if isnumeric(value)
                json_entry.(field_name) = double(value);
            elseif ischar(value) || isstring(value)
                json_entry.(field_name) = char(value);
            elseif islogical(value)
                json_entry.(field_name) = value;
            else
                json_entry.(field_name) = char(string(value));
            end
        end

        json_data = [json_data; {json_entry}];
    end

    % Write to JSON file
    fid = fopen(filename, 'w');
    fprintf(fid, '%s', jsonencode(json_data));
    fclose(fid);
end

% Run the main function if this script is executed directly
if ~isdeployed
    extract_ocr_from_docs();
end
