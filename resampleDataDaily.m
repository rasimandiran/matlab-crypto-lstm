function resampled = resampleDataDaily(data)

    % Convert the table to a timetable
    dataTT = table2timetable(data, 'RowTimes', 'open_time');
    dataTT = sortrows(dataTT, 'open_time');
    
    resampled = table();
    
    resampled.open_date = retime(dataTT, 'daily', 'first').open_time;
    
    % Apply the aggregation functions for each column
    resampled.open = retime(dataTT(:, 'open'), 'daily', 'first').open;
    resampled.high = retime(dataTT(:, 'high'), 'daily', 'max').high;
    resampled.low = retime(dataTT(:, 'low'), 'daily', 'min').low;
    resampled.close = retime(dataTT(:, 'close'), 'daily', 'last').close;
    
end