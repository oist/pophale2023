function [ev_avg,lags,ev_mat] = getEventTrigAvg(sig,event_inds,backlag,forwardlag)

min_nevents = 1; %minimum number of events where we will even compute a triggered avg


orig_size = size(sig);
%if row vector, flip
if orig_size(2) > 1 && orig_size(1) == 1
    sig = sig';
end

%convert signal into 2d array with time as first dimension
orig_size = size(sig);
if length(orig_size) > 2
    sig = reshape(sig,orig_size(1),orig_size(2)*orig_size(3));
end
[NT,p] = size(sig);


lags = (-backlag:forwardlag)';

%get rid of events that happen within the lag-range of the end points
bad_ids = find(event_inds <= backlag);
if ~isempty(bad_ids)
         fprintf('Dropping %d early events\n',length(bad_ids));
    event_inds(bad_ids) = [];
end
bad_ids = find(event_inds >= NT - forwardlag);
if ~isempty(bad_ids)
         fprintf('Dropping %d late events\n',length(bad_ids));
    event_inds(bad_ids) = [];
end

n_events = length(event_inds);

%check that we have at least the minimum number of events to work with
if n_events < min_nevents
    ev_avg = nan(length(lags),p);
    ev_std = nan(length(lags),p);
    ev_mat = nan;
    ev_cis = nan(length(lags),2);
    return
end

    ev_avg = zeros(length(lags),p);
    ev_mat = zeros(length(lags),p,numel(event_inds));
  
    for i = 1:n_events
        cur_ids = (event_inds(i)-backlag):(event_inds(i)+forwardlag);
        temp_sig = sig(cur_ids,:);
        ev_avg = ev_avg + temp_sig;
         ev_mat(:,:,i)=temp_sig;
    end
    ev_avg = ev_avg./n_events;
    
    if length(orig_size) > 2
        ev_avg = reshape(ev_avg,[length(lags) orig_size(2) orig_size(3)]);
    end
    
    ev_mat = squeeze(ev_mat)';
end