clear;
fftsize = 4096;
refractive_index_in_tissue=1.40;
resized_z = 512;

% folder of OCT scans of ~8 regions of a patient's brain
cd("D:\testingdata_vari_depth\Testing_Data\2_NW_VariDepth")
data_folder = "E:\OneDrive - Johns Hopkins\brain_cancer_oct\segmentation_data\scan_height\";
data_label = "S2_T1\"; % "cancer" or "non_cancer"
parent_folder = append(data_folder, data_label);
patient_id = '';
ofolder2=append(parent_folder, patient_id); % where the processed B-frames are stored

fheader=dir('*.header.tiff');
raw=dir('*.oct.raw');

% loop over each oct.raw file
for k=1:length(raw)
    fheaderk=fheader(k).name;
    finfo=imfinfo(fheaderk);
    
    % OCT image properties from header.tiff
    DepthSize=finfo.Width;
    LateralSize=finfo.Height;
    BytePerSample=finfo.BitsPerSample/8;
    FrameSize=DepthSize*LateralSize*BytePerSample;
    disp(['Each frame of raw data has ' num2str(DepthSize) ' (D) x ' num2str(LateralSize) ' (L) pixels.']);

    fraw=raw(k).name;
    fileInfo = dir(fraw); % struct with fields: name, folder, date, bytes, isdir, datenum
    fileSize = fileInfo.bytes;
    FrameNumber=fileSize/FrameSize;
    %fstep = 20 * FrameNumber/512; % STEPSIZE: 20 for training data, 1 for testing data
    fstep = 1;
    disp(['The raw file ' fraw ' contains ' num2str(FrameNumber) ' frames.']);
    
    startFrameNum = 0;
    fftw('planner','patient');
    str=fftw('wisdom');
    w = repmat(single(blackman(DepthSize,'periodic')),1,LateralSize);
    fid=fopen(fraw,'rb'); 
    t=startFrameNum+1:fstep:startFrameNum+FrameNumber;
    Phantomdata = zeros(2048,2048);
    
    % iterate over selected frames
    for n=1:length(t)
        display_str=sprintf('%05d',t(n));
        disp(display_str);

        fseek(fid,FrameSize*(fstep-1),'cof');
        A=fread(fid,DepthSize*LateralSize,'uint16=>single');

        A=A-single(2^15);
        RawData=reshape(A,DepthSize,LateralSize,[]);
        
        % process B-frame
        I_signal2=fft(RawData.*w,fftsize,1);
        image=abs(I_signal2(1:fftsize/2,:));
        %imm = imresize(log(image),[256 256]);
        imm = log(image);
   
        [si,ei]=regexpi(fraw,'^.+[0-9]{5}');
        ftemp=fraw(si:ei-5);
        ftemp2=fraw(si:ei-16);
        
        maxP_u16=max(max(imm));
        minP_u16=min(min(imm));

        imm=min(imm,maxP_u16);
        imm=max(imm,minP_u16);
        imm=(imm-minP_u16)/(maxP_u16-minP_u16);

        fname2=sprintf('%s\\%s%04d.oct.csv',ofolder2,ftemp,t(n));
        % writematrix(1-imm, fname2);  
        writematrix(image, fname2); % no log scaling or normalization
    end

    fclose(fid);
end
