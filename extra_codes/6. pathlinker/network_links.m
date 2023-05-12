clear all; clc; close all;

load('network.mat');

finalNetwork = string;
rowF = 1;

for rowW = 1:size(wholeNetwork,1)
    for rowO = 1:size(Omnipath,1)
        if wholeNetwork(rowW,1) == Omnipath(rowO,1) && wholeNetwork(rowW,2) == Omnipath(rowO,2)
            finalNetwork(rowF,1) = Omnipath(rowO,1);
            finalNetwork(rowF,2) = Omnipath(rowO,2);
            finalNetwork(rowF,3) = Omnipath(rowO,3);
            rowF = rowF + 1;
        end
    end
end