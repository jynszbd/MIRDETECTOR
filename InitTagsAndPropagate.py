from utils import *


# SensitiveFileLable = "FT2"
# UploadFile = "FT7"
# Nonexistentfile = "FT8"
# CrontabFile = "FS1"
# SudoersFile = "FS2"
# PasswdFile = "FS3"
# HistoryFile = "FS4"
# Deceptionlabel = "Deception"
# # DeleteLoglabel = "DeleteLog"
# DownloadFileLable = "FT1"
# # DownloadFileAndExecuteLable = "DownloadFileAndExecute"
# # FileInfectionReplacementLabel = "FileInfectionReplacement"
# # SubsystemsInitializationLable = "SubsystemsInitialization"
# # TimeBasedExecutionLabel = "TimeBasedExecution"
# # Top10ProcFileSystemAccessesLabel = "Top10ProcFileSystemAccesses"
# # Top10EtcFileSystemAccessesLabel = "Top10EtcFileSystemAccesses"
# # Top10SysFileSystemAccessLabel = "Top10SysFileSystemAccess"
# NetworkConnectionLabel = "PS1"
# SensitiveCommand = "PS6"
# ShellCommand = "PB5"



def setFileFlag(fileName):
    # sensitive_file = ["etc/shadow","ect/passwd","etc/sudoers"]
    control_file = ["/etc/crontab"]
    sudoer_file = ["/etc/sudoers"]
    passwd_file = ["/etc/passwd"]
    historical_file = [".bash_history"]
    notexist_file = ["null"]
    uploadDirectory = [
        "/phpstudy/www/DVWA/vulnerabilities/upload/hackable/uploads",
        "/phpstudy/www/DVWA/hackable/uploads",
        "/var/www/html/web3/WordPress-4.6.18/wp-content/uploads",
        "/var/lib/tomcat8/webapps/jsp/",
        "/usr/tomcat7/webapps/upload/upload/",
        "/var/www/html/uploads",
        "/www/admin/localhost_80/wwwroot/Eyoucms/",
        "/www/admin/localhost_80/wwwroot/Dedecms/",
        "/(upload)/",
        "upload"
    ]
    sensitive_file = [
        "/tmp/vUgefal",
        "/etc/group",
        "var/log/devc",
    ]
    file_dict = {}
    with open(fileName, "r") as f:
        for line in f:
            file_id, file_path = line.strip().split(": ")
            tags = []
            # if any(item in file_path for item in sensitive_file):
            #     tags.append("FT2")
            if any(item in file_path for item in control_file):
                tags.append("FH1")
            if any(item in file_path for item in sudoer_file):
                tags.append("FH2")
            if any(item in file_path for item in passwd_file):
                tags.append("FH3")
            if any(item in file_path for item in historical_file):
                tags.append("FH4")
            if any(item in file_path for item in sensitive_file):
                tags.append("FH6")
            if any(item in file_path for item in notexist_file):
                tags.append("FU3")
            if any(item in file_path for item in uploadDirectory):
                tags.append("FU1")
            if tags:
                file_dict[file_id] = tags
            else:
                file_dict[file_id] = ["unknow"]
    return file_dict
def setProcessFlag(subjectFile):
    network_connect = ["scp", "wget", "httpd"]
    shell_command = ["sh", "bash"]
    sensitive_command = \
        ["chmod", "tcpdump", "ifconfig", "sudo", "insmod", "fuser -k",
         "netstst",
         "cat /etc/hosts",
         "cat /etc/passwd",
         "cat /etc/shadow",
         "ifconfig",
         "wall -P",
         "whoami",
         "ps",
         "arp"]

    process_dict = {}
    with open(subjectFile, "r",encoding="utf-8") as f:
        for line in f:
            try:
                subject_id, subjectPropertity = line.strip().split(": ", 1)
            except:
                print(line)
            tags = []
            if any(item in subjectPropertity for item in network_connect):
                tags.append("PS1")
            if any(item in subjectPropertity for item in shell_command):
                tags.append("PB5")

            if any(item in subjectPropertity for item in sensitive_command):
                tags.append("PB3")
            if subjectPropertity != "null" and subjectPropertity != "N/A":
                tags.append("PB4")

            # if any(item in subjectPropertity for item in webshell):
            #     tags.append("PB4")
            if tags:
                process_dict[subject_id] = tags
            else:
                process_dict[subject_id] = ["unknow"]

        return process_dict

def setNetFlag(netName):
    sensitive_net = [
        "78.205.235.65",
        "200.36.109.214",
        "81.49.200.166",
        "139.123.0.113",
        "154.145.113.18",
        "61.16.39.128"
    ]
    net_dict = {}
    with open(netName, "r") as f:
        for line in f:
            net_id, net_property = line.strip().split(": ")
            tags = []

            if any(item in net_property for item in sensitive_net):
                tags.append("NT1")
            if tags:
                net_dict[net_id] = tags
            else:
                net_dict[net_id] = ["unknow"]
    return net_dict

def writeInitLabel(writename,dict):
    with open(writename,'w') as wf:
        for dictkeys in dict:
            wf.writelines(str(dictkeys)+": "+str(dict[dictkeys]))
            wf.writelines("\n")

def setProcessNet(eventName,processLabel):
    with open(eventName, "r") as f:
        for line in f:
            elements = line.split()
            source_node = elements[0]
            operation = elements[2]
            target_node = elements[3]
            target_type = elements[4]
            ##若目标的类型是socket，并且process没有标签ps1，则将process设定为NetworkConnectLabel
            if operation == "EVENT_READ" or operation == "EVENT_OPEN" or operation == "EVENT_CREATE_OBJECT":
                if target_type == "Socket" and  "PS1" not in processLabel[source_node]:
                    tags = processLabel[source_node]
                    tags.extend(["PS1"])
                    processLabel[source_node] = tags
            if operation == "EVENT_CONNECT" or operation == "EVENT_ACCEPT" :
                if "PS1" not in processLabel[source_node]:
                    tags = processLabel[source_node]
                    tags.extend(["PS1"])
                    processLabel[source_node] = tags
    return processLabel

def eventMapping(operation):
    if operation == "EVENT_READ" or operation == "EVENT_RECVMSG":
        operation_mapping_id = 0
    elif operation == "EVENT_WRITE" or operation == "EVENT_SENDMSG":
        operation_mapping_id = 1
    elif operation == "EVENT_FORK" or operation == "EVENT_CLONE":
        operation_mapping_id = 2
    elif operation == "EVENT_EXECUTE":
        operation_mapping_id = 3
    elif operation == "EVENT_CONNECT" or operation == "RECVFROM" or operation == "SENDTO":
        operation_mapping_id = 4
    elif operation == "EVENT_UNLINK" or operation == "EVENT_DELETE":
        operation_mapping_id = 5
    elif operation == "EVENT_RENAME":
        operation_mapping_id = 6
    elif operation == "EVENT_CREATE_OBJECT":
        operation_mapping_id = 7
    elif operation == "EVENT_MODIFY_FILE_ATTRIBUTES" or operation == "EVENT_UPDATE":
        operation_mapping_id = 8
    elif operation == "EVENT_EXIT" or operation == "EVENT_CLOSE":
        operation_mapping_id = 9
    elif operation == "EVENT_LOADLIBRARY":
        operation_mapping_id = 10
    elif operation == "EVENT_OPEN":
        operation_mapping_id = 11
    elif operation == "EVENT_CLOSE":
        operation_mapping_id = 12
    elif operation == "EVENT_MMAP":
        operation_mapping_id = 15
    else:
        operation_mapping_id = 16
    return operation_mapping_id



def propagate_labels(eventFile, propagationRules, processDict, fileDict,netDict,
                                                         processInitVec, fileInitVec):
    # 读取事件文件
    i = 0
    j = 0
    with open(eventFile, 'r') as f:
        events = f.readlines()
    # 解析传播规则
    rules = {}
    ##D_mapping 存储正向映射规则，R_mapping 存储反向映射规则
    ##source_label_all:统计所有的源trans的label类型，target_label_all:统计所有的目的label类型
    source_label_all = []
    target_label_all = []
    D_mapping = {}
    R_mapping = {}
    # 读取trans规则,在传输规则中，基本source_label是进程的标签，target_label是文件的标签
    with open(propagationRules, 'r') as f:
        for line in f:
            source_label, operation, target_label, direction = line.split(",")
            if source_label not in source_label_all:
                source_label_all.append(source_label)
            if target_label not in target_label_all:
                target_label_all.append(target_label)
            if direction == "D\n":
                D_mapping[(source_label, operation)] = target_label
            else:
                R_mapping[(target_label, operation)] = source_label
    # 读取事件，进行规则传输
    for event in events:
        j += 1

        elements = event.split()
        source_node = elements[0]
        operation = elements[2]
        target_node = elements[3]
        target_node_type = elements[4]
        # if target_node_type == "Socket":
        #     continue
        source_values_list = processDict[source_node]

        operation_mapping_id = eventMapping(operation)
        if operation_mapping_id == 16:
            continue
        if operation_mapping_id == 0 or operation_mapping_id == 1 or operation_mapping_id == 3 or operation_mapping_id == 10 or operation_mapping_id == 4:
            if target_node_type == "Socket":
                try:
                    target_values_list = netDict[target_node]
                except:
                    continue
                for values in target_values_list:
                    if values in target_label_all:
                        if (values, str(operation_mapping_id)) in R_mapping.keys():
                            i += 1
                            # source_node = int(source_node)
                            vec = increase_type_count(processInitVec[source_node],
                                                      R_mapping[(values, str(operation_mapping_id))])
                            processInitVec[source_node] = vec
                            # tags = processLabel[source_node]
                            # tags.extend([R_mapping[(values, str(operation_mapping_id))]])
                            # processLabel[source_node] = tags
            else:
                try:
                    target_values_list = fileDict[target_node]
                except:
                    continue
                for values in source_values_list:
                    if values in source_label_all:
                        if (values, str(operation_mapping_id)) in D_mapping.keys():
                            i += 1
                            ##假设其在D_mapping中，则要将file中添加值，添加什么值，添加D_mapping键所对应的值。
                            # target_node = int(target_node)
                            vec = increase_type_count(fileInitVec[target_node],D_mapping[(values,str(operation_mapping_id))])
                            # tags = fileLabel[target_node]
                            # tags.extend([D_mapping[(values, str(operation_mapping_id))]])
                            # fileLabel[target_node] = tags
                            fileInitVec[target_node] = vec

                for values in target_values_list:
                    if values in target_label_all:
                        if (values, str(operation_mapping_id)) in R_mapping.keys():
                            i += 1
                            # source_node = int(source_node)
                            vec = increase_type_count(processInitVec[source_node],R_mapping[(values,str(operation_mapping_id))])
                            processInitVec[source_node] = vec
                            # tags = processLabel[source_node]
                            # tags.extend([R_mapping[(values, str(operation_mapping_id))]])
                            # processLabel[source_node] = tags

        #### 如果eventType == 2 ，则对于父进程的标签除了以下标签外，全部复制给子进程
        ### PT10，PT9，PT4，PT5,PT6
        if operation_mapping_id == 2:
            for values in source_values_list:
                if values != "PB5" and values != "PB3" and values != "unknow":

                    # 将父进程的标签复制到子进程
                    i += 1
                    # target_node = int(target_node)
                    vec = increase_type_count(processInitVec[target_node],values)
                    processInitVec[target_node] = vec
                    # tags = processLabel[target_node]
                    # tags.extend([values])
                    # processLabel[target_node] = tags


    return processInitVec,fileInitVec


