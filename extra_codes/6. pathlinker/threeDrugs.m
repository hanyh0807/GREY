clear all; clc; close all;

load('3drugs_new.mat');

afaNet = string;
traNet = string;
palNet = string;

rowAF = 1;

for rowA = 1:size(Afatinib,1)
    for rowG = 1:size(GRNSNupdate,1)
        if Afatinib(rowA,1) == GRNSNupdate(rowG,1) && Afatinib(rowA,2) == GRNSNupdate(rowG,2)
            afaNet(rowAF,1) = GRNSNupdate(rowG,1);
            afaNet(rowAF,2) = GRNSNupdate(rowG,2);
            afaNet(rowAF,3) = GRNSNupdate(rowG,3);
            rowAF = rowAF + 1;
        end
    end
end

rowPF = 1;

for rowP = 1:size(Palbociclib,1)
    for rowG = 1:size(GRNSNupdate,1)
        if Palbociclib(rowP,1) == GRNSNupdate(rowG,1) && Palbociclib(rowP,2) == GRNSNupdate(rowG,2)
            palNet(rowPF,1) = GRNSNupdate(rowG,1);
            palNet(rowPF,2) = GRNSNupdate(rowG,2);
            palNet(rowPF,3) = GRNSNupdate(rowG,3);
            rowPF = rowPF + 1;
        end
    end
end

rowTF = 1;

for rowT = 1:size(Trametinib,1)
    for rowG = 1:size(GRNSNupdate,1)
        if Trametinib(rowT,1) == GRNSNupdate(rowG,1) && Trametinib(rowT,2) == GRNSNupdate(rowG,2)
            traNet(rowTF,1) = GRNSNupdate(rowG,1);
            traNet(rowTF,2) = GRNSNupdate(rowG,2);
            traNet(rowTF,3) = GRNSNupdate(rowG,3);
            rowTF = rowTF + 1;
        end
    end
end