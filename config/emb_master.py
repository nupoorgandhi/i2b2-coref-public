SAMPLE_DIR = '/projects/tir4/users/nmgandhi/coref/config/samples'

EMB_ROOT = "/projects/tir4/users/nmgandhi/coref/data/emb"
EMB_TRAIN_FILES = {100:{'src':'model=spanbert_base_partition=train_dataset=src_maxex=100.h5',
                  'tgt':'model=spanbert_base_partition=train_dataset=tgt_maxex=100.h5'},
             150: {'src': 'model=spanbert_base_partition=train_dataset=src_maxex=150.h5',
                   'tgt': 'model=spanbert_base_partition=train_dataset=tgt_maxex=150.h5'},
             200:{'src':'model=spanbert_base_partition=train_dataset=src_maxex=200.h5',
                  'tgt':'model=spanbert_base_partition=train_dataset=tgt_maxex=200.h5'},
             50:{'src':'model=spanbert_base_partition=train_dataset=src_maxex=50.h5',
                  'tgt':'model=spanbert_base_partition=train_dataset=tgt_maxex=50.h5'},
             20:{'src':'model=spanbert_base_partition=train_dataset=src_maxex=20.h5',
                  'tgt':'model=spanbert_base_partition=train_dataset=tgt_maxex=20.h5'},
             500:{'src':'model=spanbert_base_partition=train_dataset=src_maxex=500.h5',
                  'tgt':'model=spanbert_base_partition=train_dataset=tgt_maxex=500.h5'},
             }

EMB_DEV_FILES = {'src': 'model=spanbert_base_partition=dev_dataset=src_maxex=100.h5',
                 'tgt': 'model=spanbert_base_partition=dev_dataset=tgt_maxex=50.h5'}
EMB_TEST_FILES = {'src': 'model=spanbert_base_partition=test_dataset=src_maxex=100.h5',
                 'tgt': 'model=spanbert_base_partition=test_dataset=tgt_maxex=100.h5'}


INCOMPAT_PAIRS = [('ipsilateral', 'contralateral'),
                  ('superficial', 'deep'),
                  ('visceral', 'parietal'),
                  ('axial', 'abaxial'),
                  ('rostral', 'caudal'),
                  ( 'anterior', 'posterior'),
                  ('dorsal', 'ventral'),
                  ('left', 'right'),
                  ('proximal', 'distal'),
                  ('euthymia','dementia')]

COMPAT_PAIRS = [('adenocarcinoma', 'carcinoma'),
                ('birthweight', 'weight'),
                ('brachytherapy', 'therapy'),
                ('chemotherapy', 'therapy'),
                ('cystoprostatectomy', 'prostatectomy'),
                ('cytopathology', 'pathology'),
                ('empiricvancomycin', 'vancomycin'),
                ('gastrojejunostomy', 'jejunostomy'),
                ('guidewire', 'wire'),
                ('hemicolectomy', 'colectomy'),
                ('hemilaminectomy', 'laminectomy'),
                ('hemodialysis', 'dialysis'),
                ('hepatosplenomegaly', 'splenomegaly'),
                ('ischemiccardiomyopathy', 'cardiomyopathy'),
                ('ketoacidosis', 'acidosis'),
                ('levalbuterol', 'albuterol'),
                ('lymphadenopathy', 'adenopathy'),
                ('methemoglobin', 'hemoglobin'),
                ('orhydronephrosis', 'hydronephrosis'),
                ('osteoarthritic', 'arthritic'),
                ('osteochondromatosis', 'chondromatosis'),
                ('periampullary', 'ampullary'),
                ('peripancreatic', 'pancreatic'),
                ('plasmapheresis', 'pheresis'),
                ('radiotherapy', 'therapy'),
                ('serratiaurosepsis', 'sepsis'),
                ('thromboembolus', 'embolus'),
                ('urosepsis', 'sepsis')]