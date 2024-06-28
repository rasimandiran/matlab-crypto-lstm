function data = labelize(data, take_profit_ratio, stop_loss_ratio, duration)
    % Ensure the data is sorted by open_time
    data = sortrows(data, 'open_date');

    % Initialize labels
    labels = zeros(height(data), 1);
    
    % Iterate over each row in the data
    for i = 1:height(data)
        % Get the current close price
        close_price = data.close(i);

        % Calculate the take profit and stop loss thresholds
        take_profit_price = close_price * (1 + take_profit_ratio);
        stop_loss_price = close_price * (1 - stop_loss_ratio);

        % Initialize flags for labeling
        label_1 = false;
        label_2 = false;

        % Look ahead for the next N rows (duration)
        for j = i+1:min(i+duration, height(data))
            future_close_price = data.close(j);

            % Check for take profit condition
            if future_close_price > take_profit_price
                label_1 = true;
                break;
            end
            
            % Check for stop loss condition
            if future_close_price < stop_loss_price
                label_2 = true;
                break;
            end
        end

        % Assign the label based on the conditions
        if label_1
            labels(i) = 1;
        elseif label_2
            labels(i) = 2;
        else
            labels(i) = 0;
        end
    end

    data.label = labels;
end
