function s = mySilhouette(data,clusterID,randSampSize)
  
   %  https://en.wikipedia.org/wiki/Silhouette_(clustering)


      K = numel(unique(clusterID));   %# number of clusters
      
      randSamples=[];
      
      for c=1:K
          currInd=find(clusterID==c);
          rs = currInd(randsample(numel(currInd),randSampSize));
          randSamples=[randSamples;rs];
      end
      
      data=data(randSamples,:);
      clusterID=clusterID(randSamples);
      
            N = size(data,1);            %# number of instances
      
      
      %# compute pairwise distance matrix
      D = squareform( pdist(data,'euclidean'));

      
      a = zeros(N,1);
      b = zeros(N,1);
      for i=1:N
          for c=1:K
          currInd = find(clusterID==c);
          currD=D(i,currInd);
           currD(currD==0)=[]; %remove self
          meanDist(c) = mean(currD);
          end
          
          a(i) = meanDist(clusterID(i));
          meanDist(clusterID(i))=[];
          b(i) = min(meanDist);
      end
      s = (b-a) ./ max(a,b);
  end