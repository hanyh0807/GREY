clear all; clc; close all;

load('nodes.mat');

file = string;

for i = 1:10
    if i <10
        for j = 1:size(wholeNodes,1)
            if j >= 282*(i-1)+1 && j <= 282*i
                file(j,1) = wholeNodes(j,1);
                file(j,2) = 'target';
            else
                file(j,1) = wholeNodes(j,1);
                file(j,2) = 'source';
            end
        end
    else
        for j = 1:size(wholeNodes,1)
            if j >= 282*(i-1)+1 && j <= 282*i+2
                file(j,1) = wholeNodes(j,1);
                file(j,2) = 'target';
            else
                file(j,1) = wholeNodes(j,1);
                file(j,2) = 'source';
            end
        end
    end
    eval(fileName(1,1)+string(i)+fileName(1,2));
end