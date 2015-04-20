%Train Class tree

function tree = ClassTrain(Xall)

%aggregate data 
X = []; Y= [];
for k = 1:size(Xall,2)
    X = [X;cell2mat(Xall{k}(:,3))];
    Y = [Y;cell2mat(Xall{k}(:,2))];
end

%Split Data between positive and neg examples
i1 = logical(Y);
X1 = X(i1,:); Y1 = Y(i1);
X0 = X(~i1,:); Y0 = Y(~i1);

%Create Train and Test set
N0 = size(Y0,1); N1 = size(Y1,1); N = max(N0,N1);
K = 20;     %number of folds for CV
K0 = floor(N0/K);  K1 = floor(N1/K);   %size of folds 

%10 fold CV
for k = 1:K
    %indices of test data for positive and negative examples
    itest0 = false(N0,1);
    itest0((k-1)*K0+1:k*K0) = true;        
    itest1 = false(N1,1);
    itest1((k-1)*K1+1:k*K1) = true;
    
    %test and train datasets
    X0te = X0(itest0,:);   X0tr = X0(~itest0,:);
    X1te = X1(itest1,:);   X1tr = X1(~itest1,:);
      
    %bootstrap from train dataset to create equal size datasets for 0
    %and 1 examples 
    X0trb = X0(randsample(N0,round(N/2),'true'),:); 
    X1trb = X1(randsample(N1,round(N/2),'true'),:); 
    
    Xtrb = [X0trb;X1trb];   %Train Dataset
    Ytrb = [zeros(round(N/2),1); ones(round(N/2),1)];
    randind = randi(size(Xtrb,1),size(Xtrb,1),1);
    
    Xtrb = Xtrb(randind,:); Ytrb = Ytrb(randind); %randomize data order
    
    %train classification tree
    tree = fitctree(Xtrb,Ytrb);
    
    %test
    Xte = [X0te;X1te]; Yte = [zeros(size(X0te,1),1); ones(size(X1te,1),1)];
    
    Ypred = predict(tree,Xte);
    
    err(k) = sum(abs(Yte-Ypred))/(size(Yte,1)); 
    disp(err)
end

figure, plot(err)








