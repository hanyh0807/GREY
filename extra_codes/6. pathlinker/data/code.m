clear; clc; close all;

load('data.mat');

netWithSign = "";
rowIndex = 1;

for i = 1:size(Network,1)
    for j = 1:size(Omni,1)
        if Network(i,:) == Omni(j,:)
           netWithSign(rowIndex,1) = Network(i,1);
           netWithSign(rowIndex,2) = Network(i,2);
           if Sign(j,1) == 1
               netWithSign(rowIndex,3) = '+';
           elseif Sign(j,1) == -1
               netWithSign(rowIndex,3) = '-';
           end
           rowIndex = rowIndex + 1;
        end
    end
end