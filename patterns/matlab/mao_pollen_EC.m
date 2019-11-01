function [g]=mao_pollen_EC(h)
[a b]=find(h(:,:,1)==255);% assume the white value is less than 270,e.g.255.
% use h==0 if only compute for background
X=[b -a];
D=pdist(X);
D=squareform(D);
[n1 n2]=size(D);
W=zeros(n1,n2);
T=zeros(n1,n2);
[c d]=find(D<2); %4-neighbor. use D<2 if 8-neighbor
for j=1:length(c)
    W(c(j),d(j))=1;
end
T=W;

%subgraph centrality
[phi lambda]=eig(full(W));
lambda = diag(lambda);
[Val Ind]=sort(sum(phi.^2*exp(lambda),2),'descend');

%count connected component for each stage
n1=length(T);
for j=1:n1
    T(j,j)=1;
end

% g=zeros(1,20);
% for i=1:ceil(n1/120)
%     t=i*120;
%     if t>n1
%         sub=Ind;
%     else
%     sub=Ind(1:t);
%     end
%     Z=X(sub,:);
%     D2=pdist(Z);
%     e=find(D2==1);
%     D2=squareform(D2);
%     [d1 d2]=find(D2==1);
%     nb=find(Ind(d1)-Ind(d2)==1);
%     f=0;
%     for j=1:length(nb)
%         tmp=Z(d2(nb),:)-repmat(Z(d2(nb(j)),:)+[1 0],length(nb),1);
%         ff1=find(tmp(:,1)==0&tmp(:,2)==0);
%         f=f+length(ff1);
%     end
%     g(i)=length(Z)-length(e)+f;
% end
% for i=ceil(n1/120):20
%     g(i)=g(ceil(n1/120));
% end

x=0:5:100;% this is 20-dim. e.g. use 0:2:100 for 50-dim
g=x;
g(1)=0;
for i=2:length(g)
    t=round(x(i)*n1/100);
    Z=X(Ind(1:t),:);
    D2=pdist(Z);
    e=find(D2==1);
    D2=squareform(D2);
    [d1 d2]=find(D2==1);
    nb=find(Ind(d1)-Ind(d2)==1);
    f=0;
    for j=1:length(nb)
        tmp=Z(d2(nb),:)-repmat(Z(d2(nb(j)),:)+[1 0],length(nb),1);
        ff1=find(tmp(:,1)==0&tmp(:,2)==0);
        f=f+length(ff1);
    end
    g(i)=t-length(e)+f;
end
g=g(2:end);

