from dataclasses import dataclass
import json
import os
from abc import ABC
from typing import List, Tuple, Union

import requests

import logging

logger = logging.getLogger(__name__)


@dataclass
class MCTObj(ABC):
    id: str=None

    def valid(self):
        return self.id is not None


@dataclass
class MCTDataset(MCTObj):
    name: str=None
    dataset_file: str=None

    def __str__(self):
        return f'{self.id} : {self.name} \t {self.dataset_file}'


@dataclass
class MCTConceptDB(MCTObj):
    name: str=None
    conceptdb_file: str=None

    def __str__(self):
        return f'{self.id} : {self.name} \t {self.conceptdb_file}'


@dataclass
class MCTVocab(MCTObj):
    name: str=None
    vocab_file: str=None

    def __str__(self):
        return f'{self.id} : {self.vocab_file}'


@dataclass
class MCTModelPack(MCTObj):
    name: str=None
    model_pack_zip: str=None

    def __str__(self):
        return f'{self.id} : {self.name} \t {self. model_pack_zip}'


@dataclass
class MCTMetaTask(MCTObj):
    name: str=None
    
    def __str__(self):
        return f'{self.id} : {self.name}'


@dataclass
class MCTRelTask(MCTObj):
    name: str=None

    def __str__(self):
        return f'{self.id} : {self.name}'


@dataclass
class MCTUser(MCTObj):
    username: str=None

    def __str__(self):
        return f'{self.id} : {self.username}' 


@dataclass
class MCTProject(MCTObj):
    name: str=None
    description: str=None
    cuis: str=None
    dataset: MCTDataset=None
    concept_db: MCTConceptDB=None
    vocab: MCTVocab=None
    members: List[MCTUser]=None
    meta_tasks: List[MCTMetaTask]=None
    rel_tasks: List[MCTRelTask]=None

    def __str__(self):
        return f'{self.id} : {self.name} \t {self.description} \t {self.dataset}'




class MedCATTrainerSession:

    def __init__(self, server=None):
        self.username = os.getenv("MCTRAINER_USERNAME")
        password = os.getenv("MCTRAINER_PASSWORD")
        self.server = server or 'http://localhost:8001'

        payload = {"username": self.username, "password": password}
        resp = requests.post(f"{self.server}/api/api-token-auth/", json=payload)
        if 200 <= resp.status_code < 300:
            token = json.loads(resp.text)["token"]
            self.headers = {
                'Authorization': f'Token {token}',
            }
        else:
            raise MCTUtilsException(f'Failed to login to MedCATtrainer instance running at: {self.server}')

    def create_project(self, name: str, 
                       description: str,  
                       members: Union[List[MCTUser], List[str]], 
                       dataset: Union[MCTDataset, str], 
                       cuis: List[str]=[],
                       cuis_file: str=None,
                       concept_db: Union[MCTConceptDB, str]=None, 
                       vocab: Union[MCTVocab, str]=None,
                       cdb_search_filter: Union[MCTConceptDB, str]=None, 
                       modelpack: Union[MCTModelPack, str]=None, 
                       meta_tasks: Union[List[MCTMetaTask], List[str]]=[],
                       rel_tasks: Union[List[MCTRelTask], List[str]]=[]):
        """Create a new project in the MedCATTrainer session.
        Users, models, datasets etc. can be referred to by either their client wrapper object or their name, and the ID will be retrieved
        then used to create the project. Most names have a unique constraint on them so for the majority of cases will not results in an error.
        
        Only a concept_db and vocab pair, or a modelpack needs to be specified. 
        
        Setting a modelpack will also eventually automatically select meta tasks and rel tasks.

        Args:
            name (str): The name of the project.
            description (str): The description of the project.
            members (Union[List[MCTUser], List[str]]): The annotators for the project.
            dataset (Union[MCTDataset, str]): The dataset to be used in the project.
            cuis (List[str]): The CUIs to be used in the project filter.
            cuis_file (str): The file containing the CUIs to be used in the project filter, will be appended to the cuis list.
            concept_db (Union[MCTConceptDB, str], optional): The concept database to be used in the project. Defaults to None.
            vocab (Union[MCTVocab, str], optional): The vocabulary to be used in the project. Defaults to None.
            cdb_search_filter (Union[MCTConceptDB, str], optional): _description_. Defaults to None.
            modelpack (Union[MCTModelPack, str], optional): _description_. Defaults to None.
            meta_tasks (Union[List[MCTMetaTask], List[str]], optional): _description_. Defaults to None.
            rel_tasks (Union[List[MCTRelTask], List[str]], optional): _description_. Defaults to None.

        Raises:
            MCTUtilsException: If the project creation fails

        Returns:
            MCTProject: The created project
        """
        
        if all(isinstance(m, str) for m in members):
            mct_members = [u for u in self.get_users() if u.username in members]
            if len(mct_members) != len(members):
                raise MCTUtilsException(f'Not all users found in MedCATTrainer instance: {members} requested, trainer members found: {mct_members}')
            else:
                members = mct_members
            
        if isinstance(dataset, str):
            try:    
                dataset = [d for d in self.get_datasets() if d.name == dataset].pop()
            except IndexError:
                raise MCTUtilsException(f'Dataset not found in MedCATTrainer instance: {dataset}')
        
        if isinstance(concept_db, str):
            try:
                concept_db = [c for c in self.get_models()[0] if c.name == concept_db].pop()
            except IndexError:
                raise MCTUtilsException(f'Concept DB not found in MedCATTrainer instance: {concept_db}')
        
        if isinstance(vocab, str):
            try:
                vocab = [v for v in self.get_models()[1] if v.name == vocab].pop()
            except IndexError:
                raise MCTUtilsException(f'Vocab not found in MedCATTrainer instance: {vocab}')
            
        if isinstance(cdb_search_filter, str):
            try:
                cdb_search_filter = [c for c in self.get_concept_dbs() if c.name == cdb_search_filter].pop()
            except IndexError:
                raise MCTUtilsException(f'Concept DB not found in MedCATTrainer instance: {cdb_search_filter}')
        
        if isinstance(modelpack, str):
            try:
                modelpack = [m for m in self.get_model_packs() if m.name == modelpack].pop()
            except IndexError:
                raise MCTUtilsException(f'Model pack not found in MedCATTrainer instance: {modelpack}')
        
        if all(isinstance(m, str) for m in meta_tasks):
            mct_meta_tasks = [m for m in self.get_meta_tasks() if m.name in meta_tasks]
            if len(mct_meta_tasks) != len(meta_tasks):
                raise MCTUtilsException(f'Not all meta tasks found in MedCATTrainer instance: {meta_tasks} requested, trainer meta tasks found: {mct_meta_tasks}')
            else:
                meta_tasks = mct_meta_tasks
        
        if all(isinstance(r, str) for r in rel_tasks):
            mct_rel_tasks = [r for r in self.get_rel_tasks() if r.name in rel_tasks]
            if len(mct_rel_tasks) != len(rel_tasks):
                raise MCTUtilsException(f'Not all rel tasks found in MedCATTrainer instance: {rel_tasks} requested, trainer rel tasks found: {mct_rel_tasks}')
            else:
                rel_tasks = mct_rel_tasks
        
        if (concept_db or vocab) and modelpack:
            raise MCTUtilsException('Cannot specify both concept_db/vocab and modelpack')
        
        payload = {
            'name': name,
            'description': description,
            'cuis': ','.join(cuis),
            'dataset': dataset.id,
            'concept_db': concept_db.id,
            'cdb_search_filter': [cdb_search_filter.id],
            'vocab': vocab.id,
            'members': [m.id for m in members],
            'tasks': [mt.id for mt in meta_tasks],
            'relations': [rt.id for rt in rel_tasks]
        }
        if cuis_file:
            with open(cuis_file, 'rb') as f:
                resp = requests.post(f'{self.server}/api/project-annotate-entities/', data=payload, files={'cuis_file': f}, headers=self.headers)
        else:
            resp = requests.post(f'{self.server}/api/project-annotate-entities/', data=payload, headers=self.headers)
        if 200 <= resp.status_code < 300:
            resp_json = json.loads(resp.text)
            return MCTProject(id=resp_json['id'], name=name, description=description, cuis=cuis, 
                              dataset=dataset, concept_db=concept_db, vocab=vocab, members=members, 
                              meta_tasks=meta_tasks, rel_tasks=rel_tasks)
        else:
            raise MCTUtilsException(f'Failed to create project with name: {name}', resp.text)
        
    def create_dataset(self, name: str, dataset_file: str):
        resp = requests.post(f'{self.server}/api/datasets/', headers=self.headers,
                             data={'name': name},
                             files={'original_file': open(dataset_file, 'rb')})
        if 200 <= resp.status_code < 300:
            resp_json = json.loads(resp.text)
            return MCTDataset(name=name, id=resp_json['id'])
        else:
            raise MCTUtilsException(f'Failed to create dataset with name: {name}', resp.text)

    def create_user(self, username: str, password):
        payload = {
            'username': username,
            'password': password
        }
        resp = requests.post(f'{self.server}/api/users/', json=payload, headers=self.headers)
        if 200 <= resp.status_code < 300:
            resp_json = json.loads(resp.text)
            return MCTUser(username=username, id=resp_json['id'])
        else:
            raise MCTUtilsException(f'Failed to create new user with username: {username}', resp.text)

    def create_medcat_model(self, cdb:MCTConceptDB, vocab: MCTVocab):
        
        resp = requests.post(f'{self.server}/api/concept-dbs/', headers=self.headers,
                             data={'name': cdb.name},
                             files={'cdb_file': open(cdb.conceptdb_file, 'rb')})
        if 200 <= resp.status_code < 300:
            resp_json = json.loads(resp.text)
            cdb.id = resp_json['id']
        else:
            raise MCTUtilsException(f'Failed uploading MedCAT cdb model: {cdb}', resp.text)

        resp = requests.post(f'{self.server}/api/vocabs/', headers=self.headers,
                             data={'name': vocab.name},
                             files={'vocab_file': open(vocab.vocab_file, 'rb')})
        if 200 <= resp.status_code < 300:
            resp_json = json.loads(resp.text)
            vocab.id = resp_json['id']
        else:
            raise MCTUtilsException(f'Failed uploading MedCAT vocab model: {vocab}', resp.text)
    
        return cdb, vocab

    def create_medcat_model_pack(self, model_pack: MCTModelPack):
        resp = requests.post(f'{self.server}/api/modelpacks/', headers=self.headers,
                             data={'name': model_pack.name},
                             files={'model_pack': open(model_pack.model_pack_zip, 'rb')})
        if 200 <= resp.status_code < 300:
            resp_json = json.loads(resp.text)
            model_pack.id = resp_json['id']
        else:
            raise MCTUtilsException(f'Failed uploading model pack: {model_pack.model_pack_zip}', resp.text)

    def get_users(self) -> List[MCTUser]:
        users = json.loads(requests.get(f'{self.server}/api/users/', headers=self.headers).text)['results']
        return [MCTUser(id=u['id'], username=u['username']) for u in users]

    def get_models(self) -> Tuple[List[str], List[str]]:
        cdbs = json.loads(requests.get(f'{self.server}/api/concept-dbs/', headers=self.headers).text)['results']
        vocabs = json.loads(requests.get(f'{self.server}/api/vocabs/', headers=self.headers).text)['results']
        mct_cdbs = [MCTConceptDB(id=cdb['id'], name=cdb['name'], conceptdb_file=cdb['cdb_file']) for cdb in cdbs]
        mct_vocabs = [MCTVocab(id=v['id'], name=v['name'], vocab_file=v['vocab_file']) for v in vocabs]
        return mct_cdbs, mct_vocabs

    def get_model_packs(self) -> List[MCTModelPack]:
        resp = json.loads(requests.get(f'{self.server}/api/modelpacks/', headers=self.headers).text)['results']
        mct_model_packs = [MCTModelPack(id=mp['id'], name=mp['name'], model_pack_zip=mp['model_pack']) for mp in resp]
        return mct_model_packs

    def get_meta_tasks(self) -> List[MCTMetaTask]:
        resp = json.loads(requests.get(f'{self.server}/api/meta-tasks/', headers=self.headers).text)['results']
        mct_meta_tasks = [MCTMetaTask(name=mt['name'], id=mt['id']) for mt in resp]
        return mct_meta_tasks
    
    def get_rel_tasks(self) -> List[MCTRelTask]:
        resp = json.loads(requests.get(f'{self.server}/api/relations/', headers=self.headers).text)['results']
        mct_rel_tasks = [MCTRelTask(name=rt['label'], id=rt['id']) for rt in resp]
        return mct_rel_tasks
    
    def get_projects(self) -> List[MCTProject]:
        resp = json.loads(requests.get(f'{self.server}/api/project-annotate-entities/', headers=self.headers).text)['results']
        mct_projects = [MCTProject(id=p['id'], name=p['name'], description=p['description'], cuis=p['cuis'],
                                    dataset=MCTDataset(id=p['id']),
                                    concept_db=MCTConceptDB(id=p['concept_db']),
                                    vocab=MCTVocab(id=p['vocab']),
                                    members=[MCTUser(id=u) for u in p['members']],
                                    meta_tasks=[MCTMetaTask(id=mt) for mt in p['tasks']],
                                    rel_tasks=[MCTRelTask(id=rt) for rt in p['relations']]) for p in resp]
        return mct_projects
    
    def get_datasets(self) -> List[MCTDataset]:
        resp = json.loads(requests.get(f'{self.server}/api/datasets/', headers=self.headers).text)['results']
        mct_datasets = [MCTDataset(name=d['name'], dataset_file=d['original_file'], id=d['id']) for d in resp]
        return mct_datasets

    def get_project_annos(self, projects: List[MCTProject]):
        if any(p.id is None for p in projects):
            raise MCTUtilsException('One or more project.id are None and all are required to download annotations')
        
        resp = json.loads(requests.get(f'{self.server}/api/download-annos/?project_ids={",".join([str(p.id) for p in projects])}&with_text=1', 
                                       headers=self.headers).text)
        return resp



class MCTUtilsException(Exception):
    """Base exception for MedCAT Trainer API errors"""
    def __init__(self, message, original_exception=None):
        self.message = message
        self.original_exception = original_exception
        super().__init__(self.message)
    
    def __str__(self):
        return f'{self.message} \n {self.original_exception}'


if __name__ == '__main__':
    import os
    os.environ['MCTRAINER_USERNAME'] = 'admin'
    os.environ['MCTRAINER_PASSWORD'] = 'admin'

    session = MedCATTrainerSession()

    # get tests
    users = session.get_users()
    model_packs = session.get_model_packs()
    cdbs, vocabs = session.get_models() 
    meta_tasks = session.get_meta_tasks()
    datasets = session.get_datasets()
    projects = session.get_projects()

    # create tests
    # ds = session.create_dataset(name='Test DS', dataset_file='/Users/tom/phd/MedCATtrainer/notebook_docs/example_data/cardio.csv')
    cdb_file = '<model_pack_path>/cdb.dat'
    vocab_file = '<model_pack_path>/vocab.dat'
    model_pack_zip = '<model_pack_path>.zip'
    cdb, vocab = session.create_medcat_model(MCTConceptDB(name='test-cdb', conceptdb_file=cdb_file), 
                                             MCTVocab(name='test-vocab', vocab_file=vocab_file))
    session.create_medcat_model_pack(MCTModelPack(name='test-upload', model_pack_zip=model_pack_zip))
    
    cdb = cdbs[0]
    vocab = vocabs[0]
    cuis_file = '<cuis.json>' # path to a JSON formatted Array of strings for a cui filter e.g. ['C0000001', 'C0000002', 'C0000003' ... ]
    # with client wrapper objects
    # p = session.create_project(name='test-upload', description='test-upload', cuis=['C0000001'], members=[users[0]], dataset=datasets[0], concept_db=cdb, vocab=vocab, cdb_search_filter=cdb)
    
    # # directly with just names
    # p = session.create_project(name='test-upload-4', description='test-upload-2', cuis=['C0000001'], cuis_file=cuis_file, members=['admin'], 
    #                            dataset='Test DS', concept_db='test-cdb', vocab='test-vocab', cdb_search_filter=cdb, meta_tasks=['Subject', 'Status', 'Time'],
    #                            rel_tasks=['Spatial'])
    
    # get the latest created project from the latest created project
    projects = session.get_projects()
    annos = session.get_project_annos([projects[-1]])
    
