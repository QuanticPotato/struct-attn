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

-- Load the model
function loadModel()
    opt.train_perf = {}
    opt.val_perf = {}

    -- Load layers from t7 file
    print("Load layers")
    model = torch.load('entail-struct.t7')
    layers = model[1]
    model_opt = model[2]
    layer_etas = model[3]

    -- Sentence layers
    if opt.attn ~= 'none' then
        sent_encoder = layers[3]
        parser_fwd = layers[4]
        parser_bwd = layers[5]
        parser_graph1 = layers[6]
        parser_graph2 = layers[7]
    else
        sent_encoder = layers[3]
    end

    entail_layers = {}
    sent_encoder:apply(get_entail_layer) -- Fill entail_layer[sent_encoder.name]

    -- Word embeddings layers
    word_vecs_enc1 = nn.LookupTable(valid_data.source_size, opt.word_vec_size)
    word_vecs_enc2 = nn.LookupTable(valid_data.target_size, opt.word_vec_size)
    if opt.pre_word_vecs:len() > 0 then
        print("loading pre-trained word vectors")
        local f = hdf5.open(opt.pre_word_vecs)
        local pre_word_vecs = f:read('word_vecs'):all()
        for i = 1, pre_word_vecs:size(1) do
            word_vecs_enc1.weight[i]:copy(pre_word_vecs[i])
            word_vecs_enc2.weight[i]:copy(pre_word_vecs[i])
        end

        -- Valid for all models (attn, none etc)
        layers[1] = word_vecs_enc1
        layers[2] = word_vecs_enc2
    end

    if opt.gpuid >= 0 then
        for i = 1, #layers do
            layers[i]:cuda()
        end
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

    -- prototypes for sentence graphs (TODO : model_opt ?)
    print("Building graph prototypes")
    parser_context1_proto = torch.zeros(opt.max_batch_l, model_opt.max_sent_l, opt.rnn_size_parser * 2)
    parser_context2_proto = torch.zeros(opt.max_batch_l, model_opt.max_sent_l, opt.rnn_size_parser * 2)

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

    -- TODO remove
    valid_data = data.new(opt, opt.val_data_file)

    -- train(train_data, valid_data)
    loadModel()
    local score = eval(valid_data)
    print(score)
end

main()
