clear all; clc; close all;

load('GRNSN.mat');

egfr = GRNSNupdate(1,:);
erbb2 = GRNSNupdate(1,:);
cdk4 = GRNSNupdate(1,:);
cdk6 = GRNSNupdate(1,:);
map2k1 = GRNSNupdate(1,:);
map2k2 = GRNSNupdate(1,:);

egfrN = 1;
erbb2N = 1;
cdk4N = 1;
cdk6N = 1;
map2k1N = 1;
map2k2N = 1;

for i = 1:size(GRNSNupdate,1)
    if GRNSNupdate(i,1)=='EGFR'
        egfr(egfrN,:) = GRNSNupdate(i,:);
        egfrN = egfrN + 1;
    elseif GRNSNupdate(i,2)=='EGFR'
        egfr(egfrN,:) = GRNSNupdate(i,:);
        egfrN = egfrN + 1;
    end
    if GRNSNupdate(i,1)=='ERBB2'
        erbb2(erbb2N,:) = GRNSNupdate(i,:);
        erbb2N = erbb2N + 1;
    elseif GRNSNupdate(i,2)=='ERBB2'
        erbb2(erbb2N,:) = GRNSNupdate(i,:);
        erbb2N = erbb2N + 1;
    end
    if GRNSNupdate(i,1)=='CDK4'
        cdk4(cdk4N,:) = GRNSNupdate(i,:);
        cdk4N = cdk4N + 1;
    elseif GRNSNupdate(i,2)=='CDK4'
        cdk4(cdk4N,:) = GRNSNupdate(i,:);
        cdk4N = cdk4N + 1;
    end
    if GRNSNupdate(i,1)=='CDK6'
        cdk6(cdk6N,:) = GRNSNupdate(i,:);
        cdk6N = cdk6N + 1;
    elseif GRNSNupdate(i,2)=='CDK6'
        cdk6(cdk6N,:) = GRNSNupdate(i,:);
        cdk6N = cdk6N + 1;
    end
    if GRNSNupdate(i,1)=='MAP2K1'
        map2k1(map2k1N,:) = GRNSNupdate(i,:);
        map2k1N = map2k1N + 1;
    elseif GRNSNupdate(i,2)=='MAP2K1'
        map2k1(map2k1N,:) = GRNSNupdate(i,:);
        map2k1N = map2k1N + 1;
    end
    if GRNSNupdate(i,1)=='MAP2K2'
        map2k2(map2k2N,:) = GRNSNupdate(i,:);
        map2k2N = map2k2N + 1;
    elseif GRNSNupdate(i,2)=='MAP2K2'
        map2k2(map2k2N,:) = GRNSNupdate(i,:);
        map2k2N = map2k2N + 1;
    end
end