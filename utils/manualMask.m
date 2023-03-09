function manualMask(imgDir)

cd(imgDir)
imgs=dir('*.png');

%if no mask exists, make one. Then plot one on top of eachother and 

for x =1:numel(imgs)
    
    currImg= imgs(x).name;
    nameSplit=strsplit(currImg,'.');
    
   % if exist([nameSplit{1} '_mask.png'])~=2 && ~strcmp(imgs(x).name((end-7):end),'mask.png')
        
    segImg=imread(currImg);
    
    figure
    imagesc(segImg)
    axis image
    [X,Y] = getpts;
    [xq, yq]=meshgrid(1:size(segImg,2),1:size(segImg,1));
    mask = inpolygon(xq,yq,X,Y);
    
  %  imwrite(mask, 'squidMask.png')
    imwrite(mask,[nameSplit{1} '.mask.png'])
        
    close all
   %end
end






