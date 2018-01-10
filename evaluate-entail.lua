require 'nn'
require 'nngraph'
require 'hdf5'

require 'data-entail.lua'
require 'models/models-entail.lua'
require 'models/model_utils.lua'

cmd = torch.CmdLine()

-- data files
cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data_file', 'data/entail-train.hdf5', [[Path to the training *.hdf5 file]])
cmd:option('-val_data_file', 'data/entail-val.hdf5', [[Path to validation *.hdf5 file]])
cmd:option('-test_data_file', 'data/entail-test.hdf5', [[Path to test *.hdf5 file]])

cmd:option('-savefile', 'entail', [[Savefile name]])

-- model specs
cmd:option('-hidden_size', 300, [[MLP hidden layer size]])
cmd:option('-word_vec_size', 300, [[Word embedding size]])
cmd:option('-attn', 'none', [[one of {none, simple, struct}.
                              none = no intra-sentence attention (baseline model)
                              simple = simple attention model
                              struct = structured attention (syntactic attention)]])
cmd:option('-num_layers_parser', 1, [[Number of layers for the RNN parsing layer]])
cmd:option('-rnn_size_parser', 100, [[size of the RNN for the parsing layer]])
cmd:option('-use_parent', 1, [[Use soft parents]])
cmd:option('-use_children', 0, [[Use soft children]])
cmd:option('-share_params', 1, [[Share parameters between the two sentence encoders]])
cmd:option('-proj', 1, [[Have a projection layer from the Glove embeddings]])
cmd:option('-dropout', 0.2, [[Dropout probability.]])

-- optimization
cmd:option('-epochs', 100, [[Number of training epochs]])
cmd:option('-param_init', 0.01, [[Parameters are initialized over uniform distribution with support
                               (-param_init, param_init)]])
cmd:option('-optim', 'adagrad', [[Optimization method. Possible options are: 
                              sgd (vanilla SGD), adagrad, adadelta, adam]])
cmd:option('-learning_rate', 0.05, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate.]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this renormalize it
                               to have the norm equal to max_grad_norm]])
cmd:option('-pre_word_vecs', 'data/glove.hdf5', [[If a valid path is specified, then this will load 
                                      pretrained word embeddings (hdf5 file)]])
cmd:option('-fix_word_vecs', 1, [[If = 1, fix word embeddings]])
cmd:option('-max_batch_l', 32, [[If blank, then it will infer the max batch size from validation 
				   data. You should only use this if your validation set uses a different
				   batch size in the preprocessing step]])
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])
cmd:option('-print_every', 1000, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

function zero_table(t)
    for i = 1, #t do
        t[i]:zero()
    end
end

function eval(data)
    sent_encoder:evaluate()
    local nll = 0
    local num_sents = 0
    local num_correct = 0
    for i = 1, data:size() do
        print(i .. " / " .. data:size())
        local d = data[i]
        -- batch_l : Nombre de phrases dans ce batch
        -- source_l : Nombre de mots dans la phrase source
        -- target_l : Nombre de mots dans la phrase target
        -- source : Tensor de taille (batch_l x source_l)
        -- target : Tensor de taille (batch_l x target_l)
        local target, source, batch_l, target_l, source_l, label = table.unpack(d)
        local word_vecs1 = word_vecs_enc1:forward(source)
        local word_vecs2 = word_vecs_enc2:forward(target)
        if opt.attn ~= 'none' then
            set_size_encoder(batch_l, source_l, target_l,
                opt.word_vec_size, opt.hidden_size, entail_layers)
            set_size_parser(batch_l, source_l, opt.rnn_size_parser * 2, parser_layers1)
            set_size_parser(batch_l, target_l, opt.rnn_size_parser * 2, parser_layers2)

            -- resize the various temporary tensors that are going to hold contexts/grads
            local parser_context1 = parser_context1_proto[{ { 1, batch_l }, { 1, source_l } }]
            local parser_context2 = parser_context2_proto[{ { 1, batch_l }, { 1, target_l } }]

            ------ fwd prop for parser brnn for sent 1------
            -- fwd direction
            local rnn_state_parser_fwd1 = reset_state(init_parser, batch_l, 0)
            local parser_fwd_inputs1 = {}
            for t = 1, source_l do
                parser_fwd_clones[t]:evaluate()
                parser_fwd_inputs1[t] = { word_vecs1[{ {}, t }], table.unpack(rnn_state_parser_fwd1[t - 1]) }
                local out = parser_fwd_clones[t]:forward(parser_fwd_inputs1[t])
                rnn_state_parser_fwd1[t] = out
                parser_context1[{ {}, t, { 1, opt.rnn_size_parser } }]:copy(out[#out])
            end
            -- bwd direction
            local rnn_state_parser_bwd1 = reset_state(init_parser, batch_l, source_l + 1)
            local parser_bwd_inputs1 = {}
            for t = source_l, 1, -1 do
                parser_bwd_clones[t]:evaluate()
                parser_bwd_inputs1[t] = { word_vecs1[{ {}, t }], table.unpack(rnn_state_parser_bwd1[t + 1]) }
                local out = parser_bwd_clones[t]:forward(parser_bwd_inputs1[t])
                rnn_state_parser_bwd1[t] = out
                parser_context1[{ {}, t, { opt.rnn_size_parser + 1, opt.rnn_size_parser * 2 } }]:copy(out[#out])
            end

            ------ fwd prop for parser brnn for sent 2------
            -- fwd direction
            local rnn_state_parser_fwd2 = reset_state(init_parser, batch_l, 0)
            local parser_fwd_inputs2 = {}
            for t = 1, target_l do
                parser_fwd_clones[t + source_l]:training()
                parser_fwd_inputs2[t] = { word_vecs2[{ {}, t }], table.unpack(rnn_state_parser_fwd2[t - 1]) }
                local out = parser_fwd_clones[t + source_l]:forward(parser_fwd_inputs2[t])
                rnn_state_parser_fwd2[t] = out
                parser_context2[{ {}, t, { 1, opt.rnn_size_parser } }]:copy(out[#out])
            end
            -- bwd direction
            local rnn_state_parser_bwd2 = reset_state(init_parser, batch_l, target_l + 1)
            local parser_bwd_inputs2 = {}
            for t = target_l, 1, -1 do
                parser_bwd_clones[t + source_l]:training()
                parser_bwd_inputs2[t] = { word_vecs2[{ {}, t }], table.unpack(rnn_state_parser_bwd2[t + 1]) }
                local out = parser_bwd_clones[t + source_l]:forward(parser_bwd_inputs2[t])
                rnn_state_parser_bwd2[t] = out
                parser_context2[{ {}, t, { opt.rnn_size_parser + 1, opt.rnn_size_parser * 2 } }]:copy(out[#out])
            end
            parsed_context1 = parser_graph1:forward(parser_context1:contiguous())
            parsed_context2 = parser_graph2:forward(parser_context2:contiguous())
            pred_input = { word_vecs1, word_vecs2, parsed_context1, parsed_context2 }
        else
            set_size_encoder(batch_l, source_l, target_l,
                opt.word_vec_size, opt.hidden_size, entail_layers)
            pred_input = { word_vecs1, word_vecs2 }
        end
        local pred_label = sent_encoder:forward(pred_input)
        local _, pred_argmax = pred_label:max(2)
        print(pred_argmax) -- predicted label (index of label vocab)
        ---num_correct = num_correct + pred_argmax:double():view(batch_l):eq(label:double()):sum()
        ---num_sents = num_sents + batch_l
        ---nll = nll + loss
    end
    collectgarbage()
    return acc
end


function get_layer(layer)
    if layer.name ~= nil then
        if layer.name == 'word_vecs_enc2' then
            word_vecs_dec = layer
        elseif layer.name == 'parser' then
            parser = layer
        end
    end
end

function init()
        local timer = torch.Timer()
    local start_decay = 0
    params, grad_params = {}, {}
    opt.train_perf = {}
    opt.val_perf = {}

    -- Initialize layers with random weights
    print("Initializing layers with random weights")
    for i = 1, #layers do
        local p, gp = layers[i]:getParameters()
        local rand_vec = torch.randn(p:size(1)):mul(opt.param_init)
        if opt.gpuid >= 0 then
            rand_vec = rand_vec:cuda()
        end
        p:copy(rand_vec)
        params[i] = p
        grad_params[i] = gp
    end

    -- Fill word2vec layers with Glove
    if opt.pre_word_vecs:len() > 0 then
        print("loading pre-trained word vectors")
        local f = hdf5.open(opt.pre_word_vecs)
        local pre_word_vecs = f:read('word_vecs'):all()
        for i = 1, pre_word_vecs:size(1) do
            word_vecs_enc1.weight[i]:copy(pre_word_vecs[i])
            word_vecs_enc2.weight[i]:copy(pre_word_vecs[i])
        end
    end

    --copy shared params
    params[2]:copy(params[1])
    if opt.attn ~= 'none' then
        params[7]:copy(params[6])
    end

    if opt.share_params == 1 then
        if opt.proj == 1 then
            entail_layers.proj2.weight:copy(entail_layers.proj1.weight)
        end
        for k = 2, 5, 3 do
            entail_layers.f2.modules[k].weight:copy(entail_layers.f1.modules[k].weight)
            entail_layers.f2.modules[k].bias:copy(entail_layers.f1.modules[k].bias)
            entail_layers.g2.modules[k].weight:copy(entail_layers.g1.modules[k].weight)
            entail_layers.g2.modules[k].bias:copy(entail_layers.g1.modules[k].bias)
        end
    end

    -- prototypes for gradients so there is no need to clone
    word_vecs1_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.word_vec_size)
    word_vecs2_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.word_vec_size)
    sent1_context_proto = torch.zeros(opt.max_batch_l, opt.rnn_size_parser * 2)
    sent2_context_proto = torch.zeros(opt.max_batch_l, opt.rnn_size_parser * 2)
    parser_context1_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size_parser * 2)
    parser_graph1_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.word_vec_size * 2)
    parser_context2_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size_parser * 2)
    parser_graph2_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.word_vec_size * 2)

    -- clone encoder/decoder up to max source/target length
    if opt.attn ~= 'none' then
        parser_fwd_clones = clone_many_times(parser_fwd, opt.max_sent_l_src + opt.max_sent_l_targ)
        parser_bwd_clones = clone_many_times(parser_bwd, opt.max_sent_l_src + opt.max_sent_l_targ)
        for i = 1, opt.max_sent_l_src + opt.max_sent_l_targ do
            if parser_fwd_clones[i].apply then
                parser_fwd_clones[i]:apply(function(m) m:setReuse() end)
            end
            if parser_bwd_clones[i].apply then
                parser_bwd_clones[i]:apply(function(m) m:setReuse() end)
            end
        end
    end
end

function main()
    -- parse input params
    opt = cmd:parse(arg)
    if opt.gpuid >= 0 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        require 'cutorch'
        require 'cunn'
        if opt.cudnn == 1 then
            print('loading cudnn...')
            require 'cudnn'
        end
        cutorch.setDevice(opt.gpuid)
        cutorch.manualSeed(opt.seed)
    end

    -- TODO : Load layers & layers_etas from saved model.
    -- Peut etre quelques modifs pour utiliser l'embedding francais (ie les deux premieres layers).

    -- Create the data loader class.
    print('loading data...')

    train_data = data.new(opt, opt.data_file)
    valid_data = data.new(opt, opt.val_data_file)
    test_data = data.new(opt, opt.test_data_file)
    print('done!')
    print(string.format('Source vocab size: %d, Target vocab size: %d',
        valid_data.source_size, valid_data.target_size))
    opt.max_sent_l_src = valid_data.source:size(2)
    opt.max_sent_l_targ = valid_data.target:size(2)
    opt.max_sent_l = math.max(opt.max_sent_l_src, opt.max_sent_l_targ)
    if opt.max_batch_l == '' then
        opt.max_batch_l = valid_data.batch_l:max()
    end

    print(string.format('Source max sent len: %d, Target max sent len: %d',
        valid_data.source:size(2), valid_data.target:size(2)))

    -- Build model (-> global variables). These will be the model layers
    word_vecs_enc1 = nn.LookupTable(valid_data.source_size, opt.word_vec_size)
    word_vecs_enc2 = nn.LookupTable(valid_data.target_size, opt.word_vec_size)
    if opt.attn ~= 'none' then
        parser_fwd = make_lstm(valid_data, opt.rnn_size_parser, opt.word_vec_size,
            opt.num_layers_parser, opt, 'enc')
        parser_bwd = make_lstm(valid_data, opt.rnn_size_parser, opt.word_vec_size,
            opt.num_layers_parser, opt, 'enc')
        parser_graph1 = make_parser(opt.rnn_size_parser * 2)
        parser_graph2 = make_parser(opt.rnn_size_parser * 2)
        sent_encoder = make_sent_encoder(opt.word_vec_size, opt.hidden_size,
            valid_data.label_size, opt.dropout)
    else
        sent_encoder = make_sent_encoder(opt.word_vec_size, opt.hidden_size,
            valid_data.label_size, opt.dropout)
    end

    disc_criterion = nn.ClassNLLCriterion()
    disc_criterion.sizeAverage = false


    if opt.attn ~= 'none' then
        layers = {
            word_vecs_enc1, word_vecs_enc2, sent_encoder,
            parser_fwd, parser_bwd,
            parser_graph1, parser_graph2
        }
    else
        layers = { word_vecs_enc1, word_vecs_enc2, sent_encoder }
    end

    layer_etas = {}
    optStates = {}
    for i = 1, #layers do
        layer_etas[i] = opt.learning_rate -- can have layer-specific lr, if desired
        optStates[i] = {}
    end

    if opt.gpuid >= 0 then
        for i = 1, #layers do
            layers[i]:cuda()
        end
        disc_criterion:cuda()
    end

    -- these layers will be manipulated during training
    if opt.attn ~= 'none' then
        parser_layers1 = {}
        parser_layers2 = {}
        parser_graph1:apply(get_parser_layer1)
        parser_graph2:apply(get_parser_layer2)
    end
    entail_layers = {}
    -- Fill the sent_encoder tensor (ie layer). get_entail_layer just map the global array entail_layers
    sent_encoder:apply(get_entail_layer)
    if opt.attn ~= 'none' then
        if opt.cuda_mod == 1 then
            require 'cuda-mod'
            parser_layers1.dep_parser.cuda_mod = 1
            parser_layers2.dep_parser.cuda_mod = 1
        else
            if opt.attn == 'struct' then
                parser_layers1.dep_parser:double()
                parser_layers2.dep_parser:double()
            end
        end
    end
    -- train(train_data, valid_data)
    init()
    local score = eval(valid_data)
    print(score)
end

main()
