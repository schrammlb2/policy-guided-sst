# mpirun -np 6 python -u train_HER.py --env-name='FetchSlide' --n-epochs=100
# python collect_distance_data.py --episodes=50000 --env-name=FetchPickAndPlace --agent-location=saved_models/her_FetchPickAndPlace.pkl --max-steps=35
# python collect_distance_data.py --episodes=50000 --env-name=FetchSlide --agent-location=saved_models/her_FetchSlide.pkl --max-steps=35
# python train_distance_function.py  --batch-size=1024  --cuda --env-name=FetchPickAndPlace --hidden-size=256 --epoch=100 --dropout-rate=.5
# python train_distance_function.py  --batch-size=1024  --cuda --env-name=FetchSlide --hidden-size=256 --epoch=100 --dropout-rate=.5


# mpirun -np 6 python -u train_HER.py --env-name='FetchReach' --n-epochs=10 
# mpirun -np 6 python -u train_HER.py --env-name='FetchReach' --n-epochs=10 --p2p
# mpirun -np 6 python -u train_HER.py --env-name='FetchPush' --n-epochs=50 
# mpirun -np 6 python -u train_HER.py --env-name='FetchPush' --n-epochs=50 --p2p
# mpirun -np 6 python -u train_HER.py --env-name='FetchPickAndPlace' --n-epochs=200
# mpirun -np 6 python -u train_HER.py --env-name='FetchPickAndPlace' --n-epochs=200 --p2p
# mpirun -np 6 python -u train_HER.py --env-name='FetchSlide' --n-epochs=300
# mpirun -np 6 python -u train_HER.py --env-name='FetchSlide' --n-epochs=300 --p2p

# python collect_distance_data_p2p.py --env-name=FetchPickAndPlace --episodes=10000 --agent-location=saved_models/her_FetchPickAndPlace_p2p.pkl
# python train_distance_function.py --env-name=FetchPickAndPlace --epochs=200 --p2p --agent-location=saved_models/her_FetchPickAndPlace.pkl
# python collect_distance_data_p2p.py --env-name=FetchPush --episodes=10000 --agent-location=saved_models/her_FetchPush_p2p.pkl
# python train_distance_function.py --env-name=FetchPush --epochs=200 --p2p --agent-location=saved_models/her_FetchPush.pkl
# python collect_distance_data_p2p.py --env-name=FetchReach --episodes=10000 --agent-location=saved_models/her_FetchReach_p2p.pkl
# python train_distance_function.py --env-name=FetchReach --epochs=200 --p2p --agent-location=saved_models/her_FetchReach.pkl
# python collect_distance_data_p2p.py --env-name=FetchSlide --episodes=10000 --agent-location=saved_models/her_FetchReach_p2p.pkl
# python train_distance_function.py --env-name=FetchSlide --epochs=200 --p2p --agent-location=saved_models/her_FetchReach.pkl

# mpirun -np 6 python -u train_HER_mod.py --env-name='HandReach-v0' --n-epochs=50
# mpirun -np 6 python -u train_HER.py --env-name='HandManipulateBlock-v0' --n-epochs=100
# mpirun -np 6 python -u train_HER.py --env-name='HandManipulateEgg-v0' --n-epochs=100
# mpirun -np 6 python -u train_HER.py --env-name='HandManipulatePen-v0' --n-epochs=100

# mpirun -np 6 python -u train_HER_mod.py --env-name='State-Based-Navigation-2d-Map4-Goal0-v0' --n-epochs=10 --gamma=.995
# mpirun -np 6 python -u train_HER_mod.py --env-name='Limited-Range-Based-Navigation-2d-Map4-Goal0-v0' --n-epochs=4 --gamma=.995
# mpirun -np 6 python -u train_HER.py --env-name='Limited-Range-Based-Navigation-2d-Map4-Goal0-v0' --n-epochs=40 --gamma=.995
mpirun -np 3 python -u train_HER_mod.py --env-name='Limited-Range-Based-Navigation-2d-Map4-Goal0-v0' --n-epochs=5 --gamma=.995

# python main.py GymObstacle2D stable-sparse-rrt psst 