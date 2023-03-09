function recordingLength = getTotalRecordingLength(apartmentDir,filenames,fps)

recordingLength=0;
for f=1:length(filenames)
    
    fi= h5info([apartmentDir filenames{f}],'/combinedInt');
    fi.Dataspace.Size(2);
    recordingLength=recordingLength+fi.Dataspace.Size(2)/fps/3600; %Hours
end
