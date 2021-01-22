import os


TOOLKIT_CONFIG =  {'account_id':os.environ['EAI_ACCOUNT_ID'] ,
            'image': 'registry.console.elementai.com/eai.colab/ssh',
            'data': [
                        'eai.colab.public:/mnt/public',
                        ],
            'restartable':True,
            'resources': {
                'cpu': 4,
                'mem': 8,
                'gpu': 1
            },
            'interactive': False,
            'bid':9999,
            }

SLURM_CONFIG =  {'account_id':os.environ['EAI_ACCOUNT_ID'] ,
            'image': 'registry.console.elementai.com/eai.colab/ssh',
            'data': [
                        'eai.colab.public:/mnt/public',
                        ],
            'restartable':True,
            'resources': {
                'cpu': 4,
                'mem': 8,
                'gpu': 1
            },
            'interactive': False,
            'bid':9999,
            }