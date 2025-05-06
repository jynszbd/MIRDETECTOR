
from parse import *
from InitTagsAndPropagate import *
from graphBulid import *
from EncoderWLH import *
from train import *
from config import *
from FastText import *
from test import *
if __name__ == '__main__':
    train_num = 0
    prase_data = False
    InitAndProTags = False
    BulidGraph = False
    GetSubjectVector = False
    GeneratorEmbedding = False
    trainModel = False
    testModel = False
    evaluation = True
    file_train_num = 52
    file_test_num = 21
    file_evaluation_num = 17

    if prase_data:
        if train_num == 0:
        ###解析训练数据/root/baseTags/tags/rawData/drapra3/theia
            # darpaFilePath_train = r'rawData\drapra3\theia\ta1-theia-e3-official-1r.json'
            darpaFilePath_train = raw_drapra3_cadets_dir+"ta1-theia-e3-official-1r.json"
            writeSubjectFile_train = processed_train_dir+'subject.txt'
            writeFile_train = processed_train_dir+'file.txt'
            writeNetFile_train = processed_train_dir+'socket.txt'
            writeEventFile_train = processed_train_dir+'event.txt'

            print("parse train data")
            print("reading and write subject")
            subjectID,countsubject = store_subject(darpaFilePath_train,writeSubjectFile_train)
            subject_num = countsubject
            print("reading and writing file")
            fileID,countFile = store_file(darpaFilePath_train,writeFile_train,countsubject)
            print("reading and writing socket")
            netID,countsocket = store_netflow(darpaFilePath_train,writeNetFile_train,countFile)
            print("the entity num is "+str(countsocket))
            print("creat edges")
            store_event(darpaFilePath_train, subjectID, fileID, netID, writeEventFile_train)
        ##解析测试数据
        elif train_num == 1:
            darpaFilePath_test = raw_drapra3_cadets_dir+'ta1-theia-e3-official-5m.json'
            writeSubjectFile_test = processed_test_dir+'subject.txt'
            writeFile_test = processed_test_dir+'file.txt'
            writeNetFile_test = processed_test_dir+'socket.txt'
            writeEventFile_test = processed_test_dir+'event.txt'
            writeIdtouuid = processed_test_dir+'id_to_uuid.txt'
            print("parse test data")
            print("reading and write subject")
            subjectID, countsubject = store_subject(darpaFilePath_test, writeSubjectFile_test)
            subject_num = countsubject
            print("reading and writing file")
            fileID, countFile = store_file(darpaFilePath_test, writeFile_test, countsubject)
            print("reading and writing socket")
            netID, countsocket = store_netflow(darpaFilePath_test, writeNetFile_test, countFile)
            print("the entity num is " + str(countsocket))
            print("creat edges")
            store_event(darpaFilePath_test, subjectID, fileID, netID, writeEventFile_test)
            entity_mapping = writeIdToUid(writeIdtouuid,subjectID,fileID,netID)

        else:
            darpaFilePath_test = raw_drapra3_cadets_dir + 'ta1-theia-e3-official-6r.json.8'
            writeSubjectFile_test = processed_evaluation_dir + 'subject.txt'
            writeFile_test = processed_evaluation_dir + 'file.txt'
            writeNetFile_test = processed_evaluation_dir + 'socket.txt'
            writeEventFile_test = processed_evaluation_dir + 'event.txt'
            groundtruthPath = "ProcessedData/drapra3/theia/evaluation/theia_groundtruth_uuid_1.txt"
            groundtruthUuidPath = groundtruth_dir + 'theia_groundtruth_nodeId_1.txt'
            writeIdtouuid = processed_evaluation_dir + 'id_to_uuid.txt'

            print("parse test data")
            print("reading and write subject")
            subjectID, countsubject = store_subject(darpaFilePath_test, writeSubjectFile_test)
            subject_num = countsubject
            print("reading and writing file")
            fileID, countFile = store_file(darpaFilePath_test, writeFile_test, countsubject)
            print("reading and writing socket")
            netID, countsocket = store_netflow(darpaFilePath_test, writeNetFile_test, countFile)
            print("the entity num is " + str(countsocket))
            print("creat edges")
            store_event(darpaFilePath_test, subjectID, fileID, netID, writeEventFile_test)
            entity_mapping = writeIdToUid(writeIdtouuid, subjectID, fileID, netID)
            WriteGroundtruthUuid(groundtruthPath, entity_mapping, groundtruthUuidPath)
    if InitAndProTags:
        ###标签初始化和标签传递
        if train_num == 0:
            fileName = processed_train_dir+"file.txt"
            subjectFile = processed_train_dir+"subject.txt"
            eventFile = processed_train_dir+"event.txt"
            netFile = processed_train_dir + "socket.txt"
            fileDict = setFileFlag(fileName)
            processFlagDict = setProcessFlag(subjectFile)
            netDict = setNetFlag(netFile)
            processDict = setProcessNet(eventFile, processFlagDict)
            processInitVec = InitLabels_Vec(processDict)
            fileInitVec = InitLabels_Vec(fileDict)
            propagationRules = propagationRules_file
            processVecfile = processed_train_vec+"processlabelVec.txt"
            fileVecfile = processed_train_vec+"fileLabelVec.txt"

            processVecNew, fileVecNew = propagate_labels(eventFile, propagationRules, processDict, fileDict,netDict,
                                                         processInitVec, fileInitVec)
            writeInitLabel_vec(processVecfile, processVecNew)
            writeInitLabel_vec(fileVecfile, fileVecNew)
        elif train_num == 1:
            fileName = processed_test_dir+"file.txt"
            subjectFile = processed_test_dir+"subject.txt"
            eventFile = processed_test_dir+"event.txt"
            netFile = processed_test_dir+"socket.txt"
            fileDict = setFileFlag(fileName)
            processFlagDict = setProcessFlag(subjectFile)
            netDict = setNetFlag(netFile)
            processDict = setProcessNet(eventFile, processFlagDict)
            processInitVec = InitLabels_Vec(processDict)
            fileInitVec = InitLabels_Vec(fileDict)
            propagationRules = propagationRules_file
            processVecfile = processed_test_vec+"processlabelVec.txt"
            fileVecfile = processed_test_vec+"fileLabelVec.txt"
            processVecNew, fileVecNew = propagate_labels(eventFile, propagationRules, processDict, fileDict,netDict,
                                                         processInitVec, fileInitVec)
            writeInitLabel_vec(processVecfile, processVecNew)
            writeInitLabel_vec(fileVecfile, fileVecNew)
        else:
            fileName = processed_evaluation_dir + "file.txt"
            subjectFile = processed_evaluation_dir + "subject.txt"
            eventFile = processed_evaluation_dir + "event.txt"
            netFile = processed_evaluation_dir+"socket.txt"
            fileDict = setFileFlag(fileName)
            processFlagDict = setProcessFlag(subjectFile)
            netDict = setNetFlag(netFile)
            processDict = setProcessNet(eventFile, processFlagDict)
            processInitVec = InitLabels_Vec(processDict)
            fileInitVec = InitLabels_Vec(fileDict)
            propagationRules = propagationRules_file
            processVecfile = processed_evaluation_vec + "processlabelVec.txt"
            fileVecfile = processed_evaluation_vec + "fileLabelVec.txt"
            processVecNew, fileVecNew = propagate_labels(eventFile, propagationRules, processDict, fileDict,netDict,
                                                         processInitVec, fileInitVec)
            writeInitLabel_vec(processVecfile, processVecNew)
            writeInitLabel_vec(fileVecfile, fileVecNew)
    if BulidGraph:
        graphnumber = 3000
        if train_num == 0:
            readpath = processed_train_dir+"event.txt"
            writepath = train_graph+"graph.pkl"
            file_train_num = graphBuild(readpath, graphnumber, writepath)
            print("file-train-num",file_train_num)
        elif train_num == 1:
            readpath = processed_test_dir + "event.txt"
            writepath = test_graph + "graph.pkl"
            file_test_num = graphBuild(readpath, graphnumber, writepath)
            print("file-test-num", file_test_num)
        else:
            readpath = processed_evaluation_dir + "event.txt"
            writepath = evaluation_graph + "graph.pkl"
            file_evaluation_num = graphBuild(readpath, graphnumber, writepath)
            print("file-evaluation-num", file_evaluation_num)
    ###用更多的数据进行训练
    if GeneratorEmbedding:
    ###Wl图核进行生成节点邻居结构信息，并输出相对应的文件
        #WL图核生成相应的向量
        if train_num == 0:
            ###train with all graph
            ###要记住将生成的文件改个位置
            graph_path = train_graph + "graph.pkl"
            embeddingFilePath = r"ProcessedData/drapra3/cadets/train/data/data_mutil/WL_vec_d_3_128.txt"
            train_with_graph(graph_path,embeddingFilePath)
            ###test print each train graph vec
            graph_path = train_graph+"graph.pkl"
            embeddingFilePath_each = processed_train_dir+"WL_d_3_128_"
            write_all_train_graph(graph_path,embeddingFilePath_each)
        elif train_num == 1:
            graph_path = test_graph + "graph.pkl"
            embeddingFilePath = processed_test_dir + "WL_d_3_"
            write_test_graph(graph_path, embeddingFilePath)
        else:
            graph_path = evaluation_graph + "graph.pkl"
            embeddingFilePath = processed_evaluation_dir + "WL_d_3_"
            write_test_graph(graph_path, embeddingFilePath)

    ###将process节点的属性特征采用FastText进行嵌入
    if GetSubjectVector:
        if train_num == 0:
            subjectpropertypath = processed_train_dir +"subject.txt"
            cmdpath = processed_train_dir + "command.txt"
            modelpath = 'fastText.model'
            subjectpropertyvectorpath = processed_train_dir + "subject_property_vector.txt"
            FastTextTrain(subjectpropertypath,cmdpath,modelpath)
            FastTextTest(modelpath, subjectpropertypath, subjectpropertyvectorpath)
        elif train_num == 1:
            subjectpropertyPath = processed_test_dir + "subject.txt"
            modelpath = 'fastText.model'
            subjectpropertyvectorpath = processed_test_dir + "subject_property_vector.txt"
            FastTextTest(modelpath, subjectpropertyPath, subjectpropertyvectorpath)
        else:
            subjectpropertyPath = processed_evaluation_dir + "subject.txt"
            modelpath = 'fastText.model'
            subjectpropertyvectorpath = processed_evaluation_dir + "subject_property_vector.txt"
            FastTextTest(modelpath, subjectpropertyPath, subjectpropertyvectorpath)
    ###训练模型
    if trainModel:
        train(struct_size,anomalous_size,property_size,train_subject_num,alpha,beta,yam)
    if testModel:
        node_net_path = processed_test_dir + "WL_"
        process_lable_vec = processed_test_vec + "processlabelVec.txt"
        process_property_vec = processed_test_dir + "subject_property_vector.txt"

        node_net_file = wl_train_all_file
        processed_property_vec = processed_train_dir + "subject_property_vector.txt"
        process_lable_vec = processed_train_vec + "processlabelVec.txt"
        ##得到最大节点的损失值和90分位数
        max_node_threold, node_threold = getThreold( node_net_path, process_lable_vec,process_property_vec, file_test_num,
                                                    test_subject_num, alpha, beta,yam)
    if evaluation:
        modelFilePathDir = r'models_encoder'
        node_net_path = processed_test_dir + "WL_d_3_"
        process_lable_vec = processed_test_vec + "processlabelVec.txt"
        process_property_vec = processed_test_dir + "subject_property_vector.txt"
        ##得到最大节点的损失值和90分位数
        max_node_threold, node_threold = getThreold(node_net_path, process_lable_vec, process_property_vec,
                                                    file_test_num,
                                                    test_subject_num, alpha, beta, yam)
        ##得到恶意节点所在图
        groundtruthfile = r"ProcessedData/drapra3/cadets/groundtruth_nodeId.txt"
        graphfile = r"ProcessedData/drapra3/cadets/evaluation/graph/graph.pkl"
        GT_id = getGT_graph(groundtruthfile, graphfile)
        node_net_path = processed_evaluation_dir + "WL_d_3_"
        process_lable_vec = processed_evaluation_vec + "processlabelVec.txt"
        process_property_vec = processed_evaluation_dir + "subject_property_vector.txt"

        testData(modelFilePathDir, node_net_path, process_lable_vec, process_property_vec, file_evaluation_num,
                 evaluation_subject_num, alpha, beta, yam, 0.7, node_threold, GT_id)