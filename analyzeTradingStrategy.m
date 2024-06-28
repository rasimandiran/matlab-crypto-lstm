function [finalBalance, pnlAbsolute, pnlPercent, maxDrawdown, tradeCount, winRate, profitFactor, sharpeRatio] = ...
    analyzeTradingStrategy(startingBalance, closePrices, signals)
    % Initialize variables
    cash = startingBalance;
    shares = 0; % No shares initially
    tradeResults = [];
    portfolioValues = zeros(size(closePrices));
    tradeCount = 0;
    
    % Calculate returns and metrics
    for i = 1:length(signals)
        switch signals(i)
            case 1 % Buy
                if shares == 0 % Only buy if no current shares
                    shares = cash / closePrices(i); % Buy as many shares as possible with available cash
                    cash = 0; % All cash is spent
                    tradeCount = tradeCount + 1;
                end
            case 2 % Sell
                if shares > 0 % Only sell if shares are held
                    cash = shares * closePrices(i); % Convert all shares back to cash
                    tradeProfit = cash - startingBalance; % Calculate profit from this trade
                    tradeResults(end+1) = tradeProfit;
                    shares = 0; % All shares are sold
                end
        end
        
        % Update portfolio value
        if shares > 0
            portfolioValues(i) = shares * closePrices(i); % Portfolio value from shares
        else
            portfolioValues(i) = cash; % Portfolio value is just cash if no shares
        end
    end
    
    % Final balance and P&L calculations
    finalBalance = portfolioValues(end);
    pnlAbsolute = finalBalance - startingBalance;
    pnlPercent = (pnlAbsolute / startingBalance) * 100;
    
    % Maximum drawdown
    runningMax = cummax(portfolioValues);
    drawdowns = (runningMax - portfolioValues) ./ runningMax;
    maxDrawdown = max(drawdowns) * 100; % as percentage
    
    % Win rate
    wins = tradeResults(tradeResults > 0);
    winRate = (length(wins) / length(tradeResults)) * 100;  % Multiply by 100 to convert to percentage
    
    % Profit factor
    losses = tradeResults(tradeResults < 0);
    if sum(losses) ~= 0
        profitFactor = -sum(wins) / sum(losses);
    else
        profitFactor = Inf; % No losses
    end
    
    % Sharpe ratio
    dailyReturns = diff(portfolioValues) ./ portfolioValues(1:end-1);
    sharpeRatio = (mean(dailyReturns) / std(dailyReturns)) * sqrt(252); % Annualized
end