clear; clc; close all;
load('data.mat');

exp = unique(tra_exp(1:5,:));
mut = unique(tra_mut(1:4,:));

%markers = {};
seeds_exp = zeros(size(gene,1),1);
seeds_mut = zeros(size(gene,1),1);

for i = 1:size(gene,1)
    for j = 1:size(exp,1)
        if gene(i,1) == exp(j,1)
            seeds_exp(i,1) = 1;
        end
    end
end
for i = 1:size(gene,1)
    for j = 1:size(mut,1)
        if gene(i,1) == mut(j,1)
            seeds_mut(i,1) = 1;
        end
    end
end