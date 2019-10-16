%compute k-dim features based on subgraph centrality
%input: h is binary image (black and white). 
%output: g is k-dim feature

h_first= readmatrix('binary_image.txt');

h_sub = h_first(1:10,1:10)

h = rot90(fliplr(h_sub));

function [g]=mao_pollen(h)
[a b]=find(h>=255);% assume the white value is less than 270,e.g.255.
% use h==0 if only compute for background
X=[b -a];
D=pdist(X);
D=squareform(D);
[n1 n2]=size(D);
W=zeros(n1,n2);
T=zeros(n1,n2);
[c d]=find(D==1); %4-neighbor. use D<2 if 8-neighbor
for j=1:length(c)
    if double(h(c(j)))==double(h(d(j)))
        W(c(j),d(j))=1;
    else
        W(c(j),d(j))=0.01;
    end
    T(c(j),d(j))=1;
end

%subgraph centrality
[phi lambda]=eig(full(W));
lambda = diag(lambda);
[Val Ind]=sort(sum(phi.^2*exp(lambda),2),'descend');

%count connected component for each stage
n1=length(T);
for j=1:n1
    T(j,j)=1;
end

x=0:5:100;% this is 20-dim. e.g. use 0:2:100 for 50-dim
g=x;
g(1)=0;
for i=2:length(g)
    test= i
    t=round(x(i)*n1/100);
    sub=Ind(1:t);
    [p q r s]=dmperm(T(sub,sub));
    g(i)=length(r)-1;
    r2=r(1:length(r)-1);  %used for kicking out too small component
    k=find(r(2:end)-r2<1); %value 1 can be changed to other values
    g(i)=g(i)-length(k);
end
g=g(2:end);
end