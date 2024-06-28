function plotOriginalData(app)

fileName = app.ListBoxFiles.Value;
filePath = fullfile('datasets', fileName);

dataset = parquetread(filePath);
dataset = resampleDataDaily(dataset);

% get plot type from radio buttons
if app.BtnCandlestickOriginal.Value
    app.AxesOriginal.Visible = 'on';
    app.TableOriginal.Visible = 'off';
    app.TreeColumnsOriginal.Enable = 'off';
    app.SpinnerFs.Enable = 'off';

    cla(app.AxesOriginal);

    app.AxesOriginal.Title.String = 'Original Data - Candlestick';
    app.AxesOriginal.XLabel.String = "";
    app.AxesOriginal.YLabel.String = "";

    candle(app.AxesOriginal, table2timetable(dataset));
    legend(app.AxesOriginal, 'off');
    axis(app.AxesOriginal, 'auto xy');

elseif app.Btn2DPlotOriginal.Value
    app.AxesOriginal.Visible = 'on';
    app.TableOriginal.Visible = 'off';
    app.TreeColumnsOriginal.Enable = 'on';
    app.SpinnerFs.Enable = 'off';

    cla(app.AxesOriginal);
    
    app.AxesOriginal.Title.String = 'Original Data - 2D plot';
    app.AxesOriginal.XLabel.String = "";
    app.AxesOriginal.YLabel.String = "";

    checkedNodes = app.TreeColumnsOriginal.CheckedNodes;

    hold(app.AxesOriginal, 'on');

    legend_entries = {};
    for i = 1:numel(checkedNodes)
        node = checkedNodes(i);
        nodeName = node.Text;

        plot(app.AxesOriginal, dataset.open_date, dataset.(nodeName));
        legend_entries{end+1} = nodeName;
    end

    if numel(legend_entries) >= 1
        legend(app.AxesOriginal, legend_entries);
    else
        legend(app.AxesOriginal, 'off');
    end

    hold(app.AxesOriginal, 'off');
    axis(app.AxesOriginal, 'auto xy');

elseif app.BtnTableOriginal.Value
    app.AxesOriginal.Visible = 'off';
    app.TableOriginal.Visible = 'on';
    app.SpinnerFs.Enable = 'off';

    app.TableOriginal.Data = dataset;
    app.TableOriginal.ColumnName = dataset.Properties.VariableNames;


elseif app.BtnFFTOriginal.Value
    app.AxesOriginal.Visible = 'on';
    app.TableOriginal.Visible = 'off';
    app.SpinnerFs.Enable = 'on';

    cla(app.AxesOriginal);

    % if selected FFT plot type, enable Tree Columns
    app.TreeColumnsOriginal.Enable = 'on';
    app.AxesOriginal.Title.String = 'Original Data - FFT plot';
    app.AxesOriginal.XLabel.String = "f (Hz)";
    app.AxesOriginal.YLabel.String = "|fft(X)|";

    checkedNodes = app.TreeColumnsOriginal.CheckedNodes;

    hold(app.AxesOriginal, 'on');

    legend_entries = {};
    for i = 1:numel(checkedNodes)
        node = checkedNodes(i);
        nodeName = node.Text;

        Fs = app.SpinnerFs.Value;                      % Sampling frequency
        L = numel(dataset.(nodeName));  % Length of signal

        Y = fft(dataset.(nodeName));

        plot(app.AxesOriginal, Fs/L*(0:L-1), abs(Y), "LineWidth", 3);
        legend_entries{end+1} = nodeName;
    end

    if numel(legend_entries) >= 1
        legend(app.AxesOriginal, legend_entries);
    else
        legend(app.AxesOriginal, 'off');
    end

    hold(app.AxesOriginal, 'off');
    axis(app.AxesOriginal, 'auto xy');

end

end