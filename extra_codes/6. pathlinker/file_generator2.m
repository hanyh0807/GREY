clear all; clc; close all;

load('3drugs_nodes.mat');

file = string;

for i = 1:10
    if i <10
        for j = 1:size(Afatinib,1)
            if j >= 4*(i-1)+1 && j <= 4*i || j == 42 || j == 43
                file(j,1) = Afatinib(j,1);
                file(j,2) = 'target';
            else
                file(j,1) = Afatinib(j,1);
                file(j,2) = 'source';
            end
        end
    else
        for j = 1:size(Afatinib,1)
            if j >= 4*(i-1)+1 && j <= 4*i+1 || j == 42 || j == 43
                file(j,1) = Afatinib(j,1);
                file(j,2) = 'target';
            else
                file(j,1) = Afatinib(j,1);
                file(j,2) = 'source';
            end
        end
    end
    eval(fileName(1,1)+string(i)+fileName(1,4));
end

file = string;

for i = 1:9
    if i < 9
        for j = 1:size(Trametinib,1)
            if j >= 3*(i-1)+1 && j <= 3*i || j == 26 || j == 27
                file(j,1) = Trametinib(j,1);
                file(j,2) = 'target';
            else
                file(j,1) = Trametinib(j,1);
                file(j,2) = 'source';
            end
        end
    else
        for j = 1:size(Trametinib,1)
            if j >= 3*(i-1)+1 && j <= 3*i
                file(j,1) = Trametinib(j,1);
                file(j,2) = 'target';
            else
                file(j,1) = Trametinib(j,1);
                file(j,2) = 'source';
            end
        end
    end
    eval(fileName(1,2)+string(i)+fileName(1,4));
end

file = string;

for i = 1:10
    if i <10
        for j = 1:size(Palbociclib,1)
            if j >= 6*(i-1)+1 && j <= 6*i || j == 56 || j == 57
                file(j,1) = Palbociclib(j,1);
                file(j,2) = 'target';
            else
                file(j,1) = Palbociclib(j,1);
                file(j,2) = 'source';
            end
        end
    else
        for j = 1:size(Palbociclib,1)
            if j >= 6*(i-1)+1
                file(j,1) = Palbociclib(j,1);
                file(j,2) = 'target';
            else
                file(j,1) = Palbociclib(j,1);
                file(j,2) = 'source';
            end
        end
    end
    eval(fileName(1,3)+string(i)+fileName(1,4));
end