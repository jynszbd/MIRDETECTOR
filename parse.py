
import re
from tqdm import tqdm
objDict = {}
subDict = {}
allEvents = []
eventTypeSum = {}
entityTypeSum = {}
lineType = {}
i = 0  # 38000000/48000000
filelist = ['ta1-cadets-e3-official.json',
 'ta1-cadets-e3-official.json.1',
 'ta1-cadets-e3-official.json.2',
 'ta1-cadets-e3-official-1.json',
 'ta1-cadets-e3-official-1.json.1',
 'ta1-cadets-e3-official-1.json.2',
 'ta1-cadets-e3-official-1.json.3',
 'ta1-cadets-e3-official-1.json.4',
 'ta1-cadets-e3-official-2.json',
 'ta1-cadets-e3-official-2.json.1']
eventConsider = {
    'EVENT_READ',
    'EVENT_RECVMSG',
    'EVENT_WRITE',
    'EVENT_SENDMSG',
    'EVENT_FORK',
    'EVENT_CLONE',
    'EVENT_EXECUTE',
    'EVENT_CONNECT',
    'EVENT_UNLINK',
    'EVENT_RENAME',
    'EVENT_CREATE_OBJECT',
    'EVENT_MODIFY_FILE_ATTRIBUTES',
    'EVENT_LOADLIBRARY',
    'EVENT_OPEN',
    'EVENT_MMAP'

}

def store_subject(file_path,writeNamefile):
    subject_mapping = {}
    scusess_count = 0
    fail_count = 0
    subject_uuid2path = {}  #
    ##find all subject
    for file in tqdm(filelist):
        with open(file_path+file,'r',encoding='utf-8') as f:
            print("reading ")
            for line in (f):
                if "Event" in line:
                    subject_uuid = re.findall(
                        '"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}(.*?)"exec":"(.*?)"', line)
                    try:
                        #                         (subject_uuid[0][-1])
                        if subject_uuid[0][-1] :
                            subject_uuid2path[subject_uuid[0][0]] = subject_uuid[0][-1]
                            scusess_count += 1
                    except:
                        try:
                            subject_uuid2path[subject_uuid[0][0]] = "null"
                        except:
                            pass

                        fail_count += 1

    print(fail_count)
    # Store into database
    datalist = []

    j=0
    for i in subject_uuid2path.keys():
        if len(i) != 64:
            datalist.append(str(j)+": "+subject_uuid2path[i])
            subject_mapping[i] = j

            j += 1
    k = 0
    #

    # with open(writeNamefile, 'w',encoding='utf-8') as wf:
    #     wf.writelines('\n'.join(datalist))
    return subject_mapping,j

def store_file(file_path,writefile,count_subject):
    file_uuid2path={}#
    file_mapping = {}
    file_node = set()
    for file in tqdm(filelist):
        with open(file_path+file,'r',encoding='utf-8') as f:
            print("reading ")
            for line in (f):
                if "com.bbn.tc.schema.avro.cdm18.FileObject" in line:
                    Object_uuid = re.findall('FileObject":{"uuid":"(.*?)",', line)
                    try:
                        file_node.add(Object_uuid[0])

                    except:
                        print(line)
    for file in tqdm(filelist):
        with open(file_path+file,'r', encoding='utf-8') as f:
            print("reading ")
            for line in (f):
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line:
                    predicateObject_uuid = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}',
                                                      line)
                    if len(predicateObject_uuid) > 0:
                        if predicateObject_uuid[0] in file_node:
                            if '"predicateObjectPath":null,' not in line and '<unknown>' not in line:
                                path_name = re.findall('"predicateObjectPath":{"string":"(.*?)"', line)
                                file_uuid2path[predicateObject_uuid[0]] = path_name

    datalist=[]

    j = count_subject
    for i in file_uuid2path.keys():
        if file_uuid2path[i]!='none':
            datalist.append(str(j)+": "+str(file_uuid2path[i]))

            file_mapping[i] = j
            j += 1

    datalist_new=[]
    for i in datalist:
        if i[-1]!='null':
            datalist_new.append(i)
    # with open(writefile, 'w',encoding='utf-8') as wf:
    #     wf.writelines('\n'.join(datalist))

    return file_mapping,j

def store_netflow(file_path,writeFile,countFile):
    # Parse ProcessedData from logs
    netobjset = []
    netobj2hash = {}
    net_mapping = {}
    j = countFile
    for file in tqdm(filelist):
        with open(file_path+file,encoding='utf-8') as f:
            for line in f:
                if "NetFlowObject" in line:
                    try:
                        res = re.findall(
                            'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),',
                            line)[0]


                        nodeid = res[0]
                        srcaddr = res[2]
                        srcport = res[3]
                        dstaddr = res[4]
                        dstport = res[5]
                        nodeproperty = srcaddr + "," + srcport + "," + dstaddr + "," + dstport
                        netobj2hash[nodeid] = nodeproperty
                        netobjset.append(str(j)+": "+nodeproperty)

                        net_mapping[nodeid] = j
                        j += 1

                    except:
                        pass

    # with open(writeFile, 'w',encoding='utf-8') as wf:
    #    wf.writelines('\n'.join(netobjset))

    return net_mapping,j

def store_event(file_path, subjectID, fileID, netID,writeFile):
    datalist = []
    total_event_count = 0

    reverse = ["EVENT_ACCEPT", "EVENT_RECVFROM", "EVENT_RECVMSG"]
    for file in tqdm(filelist):
        with open(file_path+file,encoding='utf-8') as f:
            for line in (f):
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line and "EVENT_FLOWS_TO" not in line:
                    #                     print(line)
                    subject_uuid = re.findall('"subject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}', line)
                    predicateObject_uuid = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm18.UUID":"(.*?)"}',
                                                      line)
                    if len(subject_uuid) > 0 and len(predicateObject_uuid) > 0:
                        if subject_uuid[0] in subjectID.keys() \
                                and (predicateObject_uuid[0] in fileID.keys() or predicateObject_uuid[0] in netID.keys() or predicateObject_uuid[0] in subjectID.keys()):
                            relation_type = re.findall('"type":"(.*?)"', line)[0]
                            time_rec = re.findall('"timestampNanos":(.*?),', line)[0]
                            time_rec = int(time_rec)
                            subjectId = subjectID[subject_uuid[0]]
                            if predicateObject_uuid[0] in fileID.keys():
                                objectId = fileID[predicateObject_uuid[0]]
                                objectType = "File"
                            elif predicateObject_uuid[0] in netID.keys():
                                objectId = netID[predicateObject_uuid[0]]
                                objectType = "Socket"
                            else:
                                objectId = subjectID[predicateObject_uuid[0]]
                                objectType = "Process"
                            edge_type = relation_type
                            if edge_type not in eventConsider:
                                continue
                            edge = str(subjectId) + " " + "Process " + edge_type+" " +str(objectId) + " " + objectType +" "+ str(time_rec)
                            total_event_count += 1
                            datalist.append(edge)

    # with open(writeFile, 'w',encoding='utf-8') as wf:
    #     wf.writelines('\n'.join(datalist))
    return total_event_count

def count_event(filePath):
    type_count_event = {}
    for file in tqdm(filelist):
        with open(filePath+file, encoding='utf-8') as f:
            for line in (f):
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.Event"' in line:
                    relation_type = re.findall('"type":"(.*?)"', line)[0]
                    if relation_type not in type_count_event:
                        type_count_event[relation_type] = 1
                    else:
                        type_count_event[relation_type] += 1

    print(type_count_event)

def writeIdToUid(filepath,subjectId,fileId,netId):
    entity_mapping = {}
    entity_id_to_uuid = []
    for id in subjectId:
        entity_mapping[id] = subjectId[id]
    for id in fileId:
        entity_mapping[id] = fileId[id]
    for id in netId:
        entity_mapping[id] = netId[id]
    for id in entity_mapping:
        entity_id_to_uuid.append(str(entity_mapping[id]) + " " + str(id))
    with open(filepath,'w') as wf:
            wf.writelines('\n'.join(entity_id_to_uuid))
    return entity_mapping


def WriteGroundtruthUuid(groundtruthPath,entity_mapping,groundtruthUUidPath):
    groundtruth_mapping = []
    with open (groundtruthPath,'r') as f:
        for line in f:
            line = line.strip().split('\n')[0]
            if line in entity_mapping.keys():
                groundtruth_mapping.append(str(entity_mapping[line])+" " + line)
            else:
                print("groundtruth 不存在")
                print(line)
    with open(groundtruthUUidPath,'w') as wf:
        wf.writelines('\n'.join(groundtruth_mapping))



if __name__ == '__main__':
    # count_event(darpaFilePath)
    print("reading and write subject")
    raw_dir = "dataset/darpa/darpaData/darpa3/ta1-cadets-e3-official/"
    writeSubjectFile = "subject.txt"
    subjectID, countsubject = store_subject(raw_dir, writeSubjectFile)
    print("reading and writing file")
    writeFile = "file.txt"
    fileID, countFile = store_file(raw_dir, writeFile, countsubject)
    print("reading and writing socket")
    writeNetFile = "socket.txt"
    netID, countsocket = store_netflow(raw_dir, writeNetFile, countFile)
    print("the entity num is " + str(countsocket))
    print("creat edges")
    writeIdtouuid = "id_to_uuid.txt"
    entity_mapping = writeIdToUid(writeIdtouuid, subjectID, fileID, netID)
    writeEventFile = "event.txt"
    total_count_event = store_event(raw_dir, subjectID, fileID, netID, writeEventFile)
    print("countsubject = ", countsubject)
    print("total_node_num = ", countsocket)
    print("total_count_event = ", total_count_event)
    # groundtruthPath = "ProcessedData/drapra3/cadets/groundtruth_uuid.txt"
    # groundtruthUuidPath = "ProcessedData/drapra3/cadets/groundtruth_nodeId.txt"
    # WriteGroundtruthUuid(groundtruthPath, entity_mapping, groundtruthUuidPath)

