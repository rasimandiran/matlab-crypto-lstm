function refreshTreeItems(app)
    fileName = app.ListBoxFiles.Value;
    filePath = fullfile('datasets', fileName);
    
    data = parquetread(filePath);
    columnNames = data.Properties.VariableNames;
    
    % update tree columns for original plot
    app.TreeColumnsOriginal.Children.delete();
    
    allNode = uitreenode(app.TreeColumnsOriginal, 'Text', 'All');
    for i = 1:numel(columnNames)
        uitreenode(allNode, 'Text', columnNames{i});
    end
    % After adding all nodes, expand Tree to show all nodes
    expand(app.TreeColumnsOriginal, 'all');
end
