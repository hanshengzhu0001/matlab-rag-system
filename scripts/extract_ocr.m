% MATLAB Documentation Parser - Extract Figure Captions and Code from HTML
% This script extracts figure captions, surrounding code, and image mappings from MATLAB HTML documentation

function extract_figure_mappings_from_docs()
    % Main function to extract figure-to-caption mappings from HTML documentation

    % Configuration
    docs_root = './matlab_documents';
    output_file = 'figure_mappings.json';

    % Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif'};

    % Find all HTML files in the documentation
    fprintf('🔍 Scanning for HTML files in %s...\n', docs_root);

    html_files = {};
    html_pattern = fullfile(docs_root, '**', '*.html');
    files = dir(html_pattern);
        for i = 1:length(files)
            % Get full path relative to docs_root
            rel_path = strrep(files(i).folder, [pwd filesep docs_root], '');
            if rel_path(1) == filesep
                rel_path = rel_path(2:end);
            end
            full_rel_path = fullfile(rel_path, files(i).name);

        % Skip very small files
        if files(i).bytes < 1024
                continue;
            end

        html_files = [html_files; {fullfile(files(i).folder, files(i).name), full_rel_path}];
    end

    fprintf('📊 Found %d HTML files to process\n', size(html_files, 1));

    % Process HTML files in batches
    batch_size = 20;
    total_html = size(html_files, 1);
    figure_mappings = {};

    for start_idx = 1:batch_size:total_html
        end_idx = min(start_idx + batch_size - 1, total_html);
        fprintf('🔄 Processing batch %d-%d of %d...\n', start_idx, end_idx, total_html);

        batch_files = html_files(start_idx:end_idx, :);
        batch_results = process_html_batch(batch_files);

        figure_mappings = [figure_mappings; batch_results];

        % Save intermediate results every few batches
        if mod(end_idx, 100) == 0 || end_idx == total_html
            save_figure_mappings(figure_mappings, output_file);
            fprintf('💾 Saved intermediate results (%d figure mappings)\n', length(figure_mappings));
        end
    end

    % Final save
    save_figure_mappings(figure_mappings, output_file);
    fprintf('✅ Figure mapping extraction complete! Processed %d mappings.\n', length(figure_mappings));
end

function results = process_html_batch(html_files)
    % Process a batch of HTML files to extract figure mappings

    results = {};

    for i = 1:size(html_files, 1)
        full_path = html_files{i, 1};
        rel_path = html_files{i, 2};

        try
            % Read the HTML file
            fid = fopen(full_path, 'r', 'n', 'UTF-8');
            if fid == -1
                continue;
            end
            html_content = fread(fid, '*char')';
            fclose(fid);

            % Extract figure mappings from this HTML file
            file_mappings = extract_figure_mappings_from_html(html_content, rel_path);

            % Add to results
            results = [results; file_mappings];

        catch ME
            fprintf('⚠️  Error processing %s: %s\n', rel_path, ME.message);
        end

        % Progress indicator
        if mod(i, 5) == 0
            fprintf('  → Processed %d/%d HTML files in batch\n', i, size(html_files, 1));
        end
    end
end

function mappings = extract_figure_mappings_from_html(html_content, html_rel_path)
    % Extract figure-to-caption mappings from HTML content

    mappings = {};

    % Find all image tags with their surrounding context
    img_pattern = '<img[^>]*src="([^"]*\.(?:png|jpg|jpeg|gif))"[^>]*alt="([^"]*)"[^>]*>';
    [img_tokens, img_matches] = regexp(html_content, img_pattern, 'tokens', 'match');

    for j = 1:length(img_matches)
        img_match = img_matches{j};
        img_tokens_j = img_tokens{j};

        if length(img_tokens_j) >= 2
            img_src = img_tokens_j{1};
            img_alt = img_tokens_j{2};

            % Find surrounding code and context
            img_pos = strfind(html_content, img_match);

            if ~isempty(img_pos)
                % Extract context before image (look for code blocks)
                context_start = max(1, img_pos(1) - 1000);
                context_before = html_content(context_start:img_pos(1)-1);

                % Extract context after image
                context_end = min(length(html_content), img_pos(1) + length(img_match) + 500);
                context_after = html_content(img_pos(1) + length(img_match):context_end);

                % Find MATLAB code in context
                code_pattern = '<pre[^>]*>([^<]*)</pre>';
                code_matches = regexp([context_before context_after], code_pattern, 'tokens');

                matlab_code = '';
                if ~isempty(code_matches)
                    % Get the last code block before the image
                    for k = length(code_matches):-1:1
                        code_tokens = code_matches{k};
                        if ~isempty(code_tokens)
                            code_text = strtrim(code_tokens{1});
                            if length(code_text) > 10
                                matlab_code = code_text;
                                break;
                            end
                        end
                    end
                end

                % Find section title/context
                title_pattern = '<h[1-6][^>]*>([^<]*)</h[1-6]>';
                title_matches = regexp(context_before, title_pattern, 'tokens');

                section_title = '';
                if ~isempty(title_matches)
                    for k = length(title_matches):-1:1
                        title_tokens = title_matches{k};
                        if ~isempty(title_tokens)
                            section_title = strtrim(title_tokens{1});
                            break;
                        end
                    end
                end

                % Create mapping entry
                mapping = struct(...
                    'html_path', html_rel_path, ...
                    'image_src', img_src, ...
                    'image_alt', img_alt, ...
                    'matlab_code', matlab_code, ...
                    'section_title', section_title, ...
                    'context_before', strtrim(context_before(end-200:end)), ...
                    'context_after', strtrim(context_after(1:200)) ...
                );

                mappings = [mappings; {mapping}];
            end
        end
    end
end

function save_figure_mappings(mappings, filename)
    % Save figure mappings to JSON file

    % Convert MATLAB structs to JSON-compatible format
    json_data = {};
    for i = 1:length(mappings)
        mapping = mappings{i};

        % Ensure all fields are JSON-compatible
        json_entry = struct();
        fields = fieldnames(mapping);
        for j = 1:length(fields)
            field_name = fields{j};
            value = mapping.(field_name);

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
    extract_figure_mappings_from_docs();
end
